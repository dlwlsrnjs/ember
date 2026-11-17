"""Meaningful synthetic invariants; no repository imports or research data."""
from collections import Counter
from pathlib import Path
import hashlib,json,time
import numpy as np
from audit_controls import QueryEntry,build_control,capped_uniform_quotas

P=Path(__file__).resolve().parent
def refuses(call):
    try:call()
    except ValueError:return True
    return False
def main():
    began=time.time();q=[QueryEntry((0,9),'a',1),QueryEntry((1,9),'a',1),QueryEntry((2,9),'a',0),QueryEntry((3,9),'a',0),QueryEntry((4,8),'b',0),QueryEntry((5,8),'b',0)]
    pairs=[(0,9),(2,9),(4,8)];mask=np.array([True,False,False]);targets=np.array([[1.,.7],[0.,.2],[0.,.9]],np.float32)
    keys={row.pair:j for j,row in enumerate(q)}
    t=build_control(q,pairs,mask,'T',priorities=keys,targets=targets,slot_quotas={'a':1,'b':1},priority_contract='common_permutation')
    r=build_control(q,pairs,mask,'R',priorities=keys,targets=targets,slot_quotas={'a':1,'b':1},priority_contract='common_permutation')
    assert t.pairs==((0,9),(2,9),(4,8)) and r.pairs==((0,9),(1,9),(4,8))
    assert r.diagnostics['selected_unretained_teacher_positive_as_synthetic_negative']==1
    assert r.diagnostics['R_eligible_unretained_teacher_positive_count']==1 # RmustNOTsilentlybeRprime
    assert t.targets.tobytes()==r.targets.tobytes()==targets.tobytes()
    assert t.arrays()['y'].dtype==targets.dtype and t.arrays()['y'].shape==targets.shape
    independent={**keys,(3,9):-100}
    t2=build_control(q,pairs,mask,'T',priorities=independent,priority_contract='selector_independent')
    assert t2.pairs[1]==(3,9) and t2.pairs[0]==pairs[0]
    # Pstar/extra/ineligiblepriorityentriescanneverbecomeselectednegativepairs.
    extras={**keys,(0,9):-999,(99,99):-1000}
    rr=build_control(q,pairs,mask,'R',priorities=extras)
    assert rr.pairs[1]!=(0,9) and (99,99) not in rr.pairs
    score={row.pair:0. for row in q};score[(2,9)]=score[(3,9)]=10.;score[(5,8)]=3.
    tie={row.pair:j for j,row in enumerate(q)};tie[(3,9)]=-1
    hard=build_control(q,pairs,mask,'Hard',frozen_scores=score,tie_priorities=tie,score_provenance='fixture:frozen_training_checkpoint')
    assert hard.pairs==((0,9),(3,9),(5,8))
    # Explicitnewallocationchangesnegativebucketassignmentalone.
    twoa=[(0,9),(2,9),(3,9)]
    moved=build_control(q,twoa,mask,'T',priorities=keys,slot_buckets=('a','a','b'),slot_quotas={'a':1,'b':1})
    assert moved.pairs[0]==twoa[0] and moved.pairs[2]==(4,8)
    assert refuses(lambda:build_control(q,twoa,mask,'T',priorities=keys,slot_quotas={'a':1,'b':1}))
    bad=[]
    bad.append(refuses(lambda:build_control(q+q[:1],pairs,mask,'T',priorities=keys)))
    bad.append(refuses(lambda:build_control(q,[(0,9),(2,9),(2,9)],mask,'R',priorities=keys)))
    bad.append(refuses(lambda:build_control(q,[(0,9),(99,99),(4,8)],mask,'R',priorities=keys)))
    bad.append(refuses(lambda:build_control(q,pairs,np.array([1,0,0]),'R',priorities=keys)))
    bad.append(refuses(lambda:build_control(q,pairs,mask,'R',priorities={})))
    bad.append(refuses(lambda:build_control(q,pairs,mask,'R',priorities=keys,targets=np.zeros(3))))
    bad.append(refuses(lambda:build_control(q,[(2,9),(3,9),(4,8)],mask,'R',priorities=keys)))
    bad.append(refuses(lambda:build_control(q,[(0,9),(2,9),(3,9),(4,8)],np.array([True,False,False,False]),'T',priorities=keys,slot_buckets=('a','a','a','a'))))
    bad.append(refuses(lambda:build_control(q,pairs,mask,'Hard',frozen_scores=score,tie_priorities=tie)))
    bad.append(refuses(lambda:build_control(q,pairs,mask,'Hard',frozen_scores={**score,(2,9):float('nan')},tie_priorities=tie,score_provenance='frozen')))
    assert all(bad)
    # NonstrictnumericIDsareunnecessary: arbitraryhashableIDsworkwithQorderfallback.
    textq=[QueryEntry(('reader','article0'),'bucket',1),QueryEntry(('other','article1'),'bucket',0)]
    text=build_control(textq,[('reader','article0'),('other','article1')],mask[:2],'T',priorities={('other','article1'):b'abc'})
    assert text.pairs[0]==('reader','article0')
    assert build_control([],[],np.array([],bool),'R',priorities={}).pairs==()
    # Efficientwaterfillmustexactlymatchthetransparentiterativealgorithm.
    rng=np.random.default_rng(9501);watercases=0
    for n in range(12):
        for repeat in range(30):
            cap={j:int(v) for j,v in enumerate(rng.integers(0,30,n))};order=list(rng.permutation(n));budget=int(rng.integers(0,sum(cap.values())+1))
            got=capped_uniform_quotas(cap,budget,order);expected={b:0 for b in order};left=budget
            while left:
                for b in order:
                    if left and expected[b]<cap[b]:expected[b]+=1;left-=1
            assert got==expected;watercases+=1
    assert refuses(lambda:capped_uniform_quotas({'a':1},2,['a']))
    assert refuses(lambda:capped_uniform_quotas({'a':1},1,['b']))
    assert refuses(lambda:capped_uniform_quotas({'a':True},1,['a']))
    assert refuses(lambda:capped_uniform_quotas({'a':1},-1,['a']))
    # PrioritytiesaredeterministicunderfixedQ;explicitcaller tiekeyscontrolthem.
    tied={row.pair:0 for row in q};priority_tie={row.pair:-j for j,row in enumerate(q)}
    tied_result=build_control(q,pairs,mask,'T',priorities=tied,tie_priorities=priority_tie)
    assert tied_result.pairs==((0,9),(3,9),(5,8))
    report={'all_passed':True,'waterfill_vs_iterative_cases':watercases,'invalid_control_contract_cases_rejected':len(bad),
        'R_is_Q_minus_retained_Pstar_not_all_teacherpositives':True,'Pstar_targets_dtype_slotorder_exact':True,
        'explicit_common_and_independent_priorities':True,'hard_frozen_score_and_tieorder':True,'capacity_uniqueness_ineligible_pair_guards':True,
        'arbitrary_hashable_IDs_and_zero_budget':True,'actual_research_inputs_or_models_used':False,
        'source_sha256':hashlib.sha256((P/'audit_controls.py').read_bytes()).hexdigest(),'seconds':time.time()-began}
    (P/'SYNTHETIC_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print('R95 SYNTHETIC PASS',watercases,'waterfillcases',len(bad),'rejectedcontracts',flush=True)

if __name__=='__main__':main()
