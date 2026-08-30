from pathlib import Path
import hashlib
import numpy as np
from bdse.planner.tournament import _signed_attribution_spectrum,_icer_attribution_resolved_feature_matrix,_icer_local_regret_lower_bound


def _memory(path:Path, y, kind):
    X=np.zeros((64,1),np.float32); y=np.asarray(y,np.float32); np.savez_compressed(path,memory_metric_z=X,teacher_improvement=y,feature_mean=np.zeros(1,np.float32),feature_std=np.ones(1,np.float32),feature_names=np.asarray(['x'],dtype='U8'),feature_metric_weight=np.ones(1,np.float32),neighbor_k_values=np.asarray([32,64],np.int32),se_multiplier=np.asarray([1.0],np.float32),certificate_kind=np.asarray([kind],dtype='U64'),downside_multiplier=np.asarray([1.0],np.float32)); return hashlib.sha256(path.read_bytes()).hexdigest()

def test_full_signed_spectrum_retains_all_budget_entries():
    x=np.arange(1,17,dtype=float); x[1::2]*=-1; s=_signed_attribution_spectrum(x); assert s.shape==(16,); assert np.count_nonzero(s)==16; assert np.isclose(np.abs(s).sum(),1.0); assert abs(s[0])>=abs(s[-1])

def test_attribution_resolved_matrix_has_candidate_and_delta_spectra():
    c=np.array([[0.,1.,2.],[0.,3.,1.]],float); m,n=_icer_attribution_resolved_feature_matrix(c,np.ones(3,bool),0,1); assert m.shape==(3,32); assert len(n)==32; assert np.isclose(np.abs(m[2,:16]).sum(),1); assert np.isclose(np.abs(m[2,16:]).sum(),1)

def test_downside_certificate_rejects_positive_mean_with_large_negative_tail(tmp_path:Path):
    # Positive mean, but downside RMS exceeds it. Mean-SE is intentionally less tail-sensitive.
    y=np.r_[np.full(63,.05),-1.0]
    p=tmp_path/'m.npz'; h=_memory(p,y,'mean_minus_downside_rms'); s=_icer_local_regret_lower_bound(np.zeros((1,1)),['x'],str(p),h); assert s[0]<0

def test_meanse_memory_backward_semantics(tmp_path:Path):
    y=np.full(64,.1); p=tmp_path/'m.npz'; h=_memory(p,y,'mean_minus_standard_error'); s=_icer_local_regret_lower_bound(np.zeros((1,1)),['x'],str(p),h); assert s[0]>0

def test_attribution_resolved_risk_feature_view_has_exact_group_schema():
    from bdse.planner.tournament import _icer_regret_risk_feature_matrix, _DACER_FEATURE_NAMES, _ICER_ATTRIBUTION_RESOLVED_FEATURE_NAMES
    feat=np.zeros((3,len(_DACER_FEATURE_NAMES)),float); tr=np.zeros((3,0),float); ar=np.zeros((3,32),float)
    x,n=_icer_regret_risk_feature_matrix(feat,list(_DACER_FEATURE_NAMES),tr,[],"attribution_resolved",ar,list(_ICER_ATTRIBUTION_RESOLVED_FEATURE_NAMES))
    assert x.shape==(3,50); assert len(n)==50; assert n[0].startswith('evidence::'); assert n[-1].startswith('attribution::delta_atom_signed_spectrum_')
