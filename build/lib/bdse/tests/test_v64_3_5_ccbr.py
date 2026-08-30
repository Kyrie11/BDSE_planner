from __future__ import annotations

from pathlib import Path

import torch

from bdse.config import load_config
from bdse.model.bdse_model import BDSEModel, _CompleteCandidateBoundaryRouter
from bdse.model.losses import _exact_winner_flip_critical_proposal_loss
from bdse.tools.validate_v64_pipeline_config import _check


def _model() -> BDSEModel:
    return BDSEModel(load_config('bdse/configs/v64_3_5_cc_aocc_ccbr_lea_daepc_screen_2gpu.yaml'))


def test_ccbr_is_exact_step_zero_noop_and_full_support() -> None:
    model = _model()
    adapter = model.critical_proposal_adapter
    assert isinstance(adapter, _CompleteCandidateBoundaryRouter)
    assert torch.count_nonzero(adapter.residual_head[-1].weight).item() == 0
    h = model.hidden_dim
    B,E,K=2,7,32
    residual,winner,flip=adapter(
        torch.randn(B,E,5*h), torch.randn(B,K,h), torch.ones(B,K,dtype=torch.bool),
        torch.randn(B,K), torch.rand(B,K),
    )
    assert residual.shape==(B,E)
    assert winner.shape==(B,E,K)
    assert flip.shape==(B,E,K)
    assert torch.count_nonzero(residual).item()==0
    assert torch.isfinite(winner).all() and torch.isfinite(flip).all()


def test_ccbr_action_permutation_equivariance() -> None:
    model=_model(); adapter=model.critical_proposal_adapter
    assert isinstance(adapter,_CompleteCandidateBoundaryRouter)
    with torch.no_grad():
        torch.nn.init.normal_(adapter.residual_head[-1].weight,std=0.01)
    h=model.hidden_dim; B,E,K=1,4,8
    atom=torch.randn(B,E,5*h); action=torch.randn(B,K,h)
    valid=torch.tensor([[True,True,False,True,True,True,False,True]])
    cost=torch.randn(B,K); rank=torch.rand(B,K)
    perm=torch.tensor([5,0,7,2,1,6,4,3])
    r1,w1,f1=adapter(atom,action,valid,cost,rank)
    r2,w2,f2=adapter(atom,action[:,perm],valid[:,perm],cost[:,perm],rank[:,perm])
    torch.testing.assert_close(r1,r2,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(w1[:,:,perm],w2,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(f1[:,:,perm],f2,rtol=1e-5,atol=1e-6)


def test_lea_supervises_exact_winner_and_flip_endpoints_only() -> None:
    J0=torch.zeros((1,3)); g=torch.zeros((1,2,3))
    valid=torch.ones((1,3),dtype=torch.bool); active=torch.ones((1,2),dtype=torch.bool)
    proposal=torch.zeros((1,2)); deployment=torch.tensor([[True,False]])
    target=torch.tensor([0]); costs=torch.ones((1,2))
    teacher_cost=torch.tensor([[0.0,0.5,0.4]])
    teacher_g=torch.tensor([[[-0.6,0.0,0.0],[0.0,0.0,0.0]]])
    win_logits=torch.zeros((1,2,3)); flip_logits=torch.zeros((1,2,3))
    cfg={'training':{'exact_winner_flip_criticality':{
        'enabled':True,'target_source':'teacher_interface','positive_weight':1.0,'negative_weight':1.0,
        'rank_weight':0.0,'pairwise_rank_weight':0.0,'coverage_weight':0.0,'exchange_rank_weight':0.0,
        'adapter_residual_alignment_weight':0.0,'boundary_attribution_weight':0.0,
        'endpoint_attribution_weight':1.0,'endpoint_attribution_severity_weight':0.0,
    }}}
    result=_exact_winner_flip_critical_proposal_loss(
        J0,g,valid,active,proposal,deployment,target,costs,cfg,
        teacher_cost=teacher_cost,teacher_g=teacher_g,
        critical_winner_endpoint_logits=win_logits,critical_flip_endpoint_logits=flip_logits,
        return_adapter_diagnostic=True,
    )
    loss,recall,crit_frac,scene,aligned,acra,lba,bound_rep,lea,endpoint_rep=result
    assert torch.isfinite(loss)
    assert float(recall)==1.0 and float(crit_frac)==0.5 and float(scene)==1.0 and float(aligned)==1.0
    assert float(acra)==0.0 and float(lba)==0.0 and float(bound_rep)==0.0
    assert float(lea)>0.0 and float(endpoint_rep)==1.0


def test_v64_3_5_ccbr_contracts_pass() -> None:
    for name in ('v64_3_5_cc_aocc_ccbr_lea_daepc_train_2gpu.yaml','v64_3_5_cc_aocc_ccbr_nolea_daepc_train_2gpu.yaml'):
        r=_check(Path('bdse/configs')/name,'train','v64.3.5')
        assert r['pass'],r['failures']
    e=_check(Path('bdse/configs/v64_3_5_cc_aocc_ccbr_cl.yaml'),'eval','v64.3.5')
    assert e['pass'],e['failures']


def test_screen_checker_distinguishes_nolba_and_nolea_tokens() -> None:
    from bdse.tools.check_v64_3_3_acquisition_screen import build_report
    base = {
        'critical_adapter_parameter_delta_rms': 1e-3,
        'critical_proposal_residual_rms': 0.1,
        'L_critical_adapter_residual_alignment': 0.1,
        'val_teacher_exact_winner_flip_critical_recall_topm_micro': 0.36,
        'val_teacher_exact_winner_flip_critical_recall_topm': 0.31,
        'val_teacher_exact_winner_flip_critical_recall_selected_micro': 0.26,
        'val_teacher_exact_winner_flip_critical_scene_rate': 0.3,
        'val_teacher_exact_winner_flip_critical_count': 0.5,
        'val_proposal_decisive_atom_recall': 0.79,
        'val_teacher_action_match': 0.26,
    }
    rows=[dict(base,epoch=-1),dict(base,epoch=0)]
    nolba=build_report(rows,'FPCCA-noLBA+ACRA-full-support')
    nolea=build_report(rows,'CCBR-noLEA+ACRA-full-support')
    assert nolba['lba_required'] is False
    assert nolea['lea_required'] is False
