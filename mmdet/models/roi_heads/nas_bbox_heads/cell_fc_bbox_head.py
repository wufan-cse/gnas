# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F

from mmdet.registry import MODELS

from ..bbox_heads.bbox_head import BBoxHead
from .nas_utils.search_arch import Cell as SearchCell
from .nas_utils.augment_arch import Cell as AugmentCell
from .nas_utils.genotypes import PRIMITIVES
from .nas_utils.genotypes import Genotype


@MODELS.register_module()
class NASConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                            /-> cls fcs -> cls
        shared convs -> shared fcs -> cells 
                                            \-> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 0,
                 num_cells: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_fcs: int = 0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 is_search: bool = True,
                 arch_path: str = None,
                 cell_cfg: Optional[Union[dict, ConfigDict]] = None,
                 conv_cfg: Optional[Union[dict, ConfigDict]] = None,
                 norm_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cells +
                num_cls_fcs + num_reg_fcs > 0)
                
        # assert is_search is False and arch_path is None
                
        #if num_cls_cells > 0 or num_reg_convs > 0:
        #    assert num_shared_fcs == 0
        #if not self.with_cls:
        #    assert num_cls_cells == 0 and num_cls_fcs == 0
        #if not self.with_reg:
        #    assert num_reg_convs == 0 and num_reg_fcs == 0
            
        self.is_search = is_search
        self.arch_path = arch_path

        self.num_cells = num_cells

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        if cell_cfg is None:
            self.cell_cfg = {'C': 36, 'steps': 4, 'multiplier': 4, 'stem_multiplier': 3, 'drop_prob': 0.0}
        else:
            self.cell_cfg = cell_cfg
            
        self._initialize_alphas(is_search)
        
        # TODO 256 dim
        self.cells, self.fcs, self.cls_last_dim = \
            self._add_cell_fc_branch(
                self.num_cls_cells, self.num_cls_fcs, 64, is_search=is_search, cell_cfg=self.cell_cfg)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]
            

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim
        
    def _add_cell_fc_branch(self,
                            num_branch_cells: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False,
                            is_search: bool = True,
                            cell_cfg: dict = None) -> tuple:
        """Add shared or separable branch.

        stem -> cells -> nn.Conv2d -> avg pool (optional) -> fcs
        """
        C_curr = cell_cfg['stem_multiplier'] * cell_cfg['C']
        self.stem = nn.Sequential(
          nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
          nn.BatchNorm2d(C_curr))
     
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, in_channels
        reduction_prev = False
        
        # add branch specific conv layers
        branch_cells = nn.ModuleList()
        if num_branch_cells > 0:
            for i in range(num_branch_cells):
                if i in [num_branch_cells // 3, 2 * num_branch_cells // 3]:
                    C_curr *= 2
                    reduction = True
                else:
                    reduction = False
                
                if is_search:
                    CellModule = SearchCell(cell_cfg['steps'], cell_cfg['multiplier'], C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                else:
                    CellModule = AugmentCell(self.genotype(), C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                
                reduction_prev = reduction
                branch_cells.append(CellModule)
                C_prev_prev, C_prev = C_prev, cell_cfg['multiplier'] * C_curr
            
            self.align_conv = nn.Conv2d(C_prev, self.conv_out_channels, 3, padding=1, bias=False)
            last_layer_dim = self.conv_out_channels
            
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
            
        return branch_cells, branch_fcs, last_layer_dim
        
    def _initialize_alphas(self, is_search):
        k = sum(1 for i in range(self.cell_cfg['steps']) for n in range(2+i))
        num_ops = len(PRIMITIVES)
    
        if is_search:
            self.alphas_normal = torch.nn.Parameter(Variable(1e-3*torch.randn(k, num_ops)), requires_grad=True)
            self.alphas_reduce = torch.nn.Parameter(Variable(1e-3*torch.randn(k, num_ops)), requires_grad=True)
        else:
            augment_arch = torch.load(self.arch_path)
            alphas_normal = augment_arch['state_dict']['roi_head.bbox_head.alphas_normal']
            alphas_reduce = augment_arch['state_dict']['roi_head.bbox_head.alphas_reduce']
            
            self.alphas_normal = Variable(alphas_normal, requires_grad=False)
            self.alphas_reduce = Variable(alphas_reduce, requires_grad=False)
            
        # self._arch_parameters = [
        #     self.alphas_normal,
        #     self.alphas_reduce,
        # ]
            
    def genotype(self):
    
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.cell_cfg['steps']):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
    
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    
        steps = self.cell_cfg['steps'] 
        multiplier = self.cell_cfg['multiplier']
        
        concat = range(2+steps-multiplier, steps+2)
        genotype = Genotype(
          normal=gene_normal, normal_concat=concat,
          reduce=gene_reduce, reduce_concat=concat
        )
        
        return genotype

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # TODO
        s0 = s1 = self.stem(torch.reshape(x, (-1, 64, 4, 4)))
        for i, cell in enumerate(self.cls_cells):
            if self.is_search:
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                    
                s0, s1 = s1, cell(s0, s1, weights)
            else:
                s0, s1 = s1, cell(s0, s1, self.cell_cfg['drop_prob'])

        # separate branches
        x = s1
        x = self.align_conv(x)

        if x.dim() > 2:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
        for fc in self.cls_fcs:
            x = self.relu(fc(x))

        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred
        
        
@MODELS.register_module()
class NASBBoxHead(NASConvFCBBoxHead):
    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_cells=2,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
