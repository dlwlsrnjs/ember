"""Fixed-budget synthetic-feedback controls. Python stdlib + NumPy only.

This is a data-construction utility, not a new learning/selection method.
All randomness/ranker scores enter explicitly as caller-provided mappings.
R means Q minus RETAINED positive pairs, not all teacher-positive pairs.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass
from typing import Hashable,Mapping,Sequence
import math
import numpy as np

Pair=tuple[Hashable,Hashable]

@dataclass(frozen=True)
class QueryEntry:
    pair:Pair
    bucket:Hashable
    teacher_label:int

@dataclass(frozen=True)
class ControlResult:
    pairs:tuple[Pair,...]
    targets:np.ndarray
    diagnostics:dict

    def arrays(self,id_dtype=np.int64):
        """Integer-ID convenience export; targets preserve caller dtype/shape."""
        return {'u':np.asarray([p[0] for p in self.pairs],dtype=id_dtype),
                'i':np.asarray([p[1] for p in self.pairs],dtype=id_dtype),'y':self.targets.copy()}

def _integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
        raise ValueError(name+' must be a nonnegative integer')
    return int(value)

def capped_uniform_quotas(capacities:Mapping[Hashable,int],budget:int,tie_order:Sequence[Hashable]):
    """Capped integer max-min allocation with explicit deterministic remainder.

    Independent direction budgets should be allocated in separate calls.
    This function does not choose buckets or budgets from outcome utilities.
    """
    cap={b:_integer(v,'capacity') for b,v in capacities.items()};budget=_integer(budget,'budget')
    order=tuple(tie_order)
    if len(order)!=len(set(order)) or set(order)!=set(cap):raise ValueError('tie_order must contain each capacity bucket exactly once')
    if budget>sum(cap.values()):raise ValueError('budget exceeds total capacity')
    lo,hi=0,max(cap.values(),default=0)
    while lo<hi:
        mid=(lo+hi+1)//2
        if sum(min(v,mid) for v in cap.values())<=budget:lo=mid
        else:hi=mid-1
    result={b:min(cap[b],lo) for b in order};left=budget-sum(result.values())
    for b in order:
        if left and result[b]<cap[b]:result[b]+=1;left-=1
    assert left==0 and sum(result.values())==budget
    return result

def _entries(entries):
    records=[];lookup={};position={}
    for row in entries:
        if not isinstance(row,QueryEntry):
            try:row=QueryEntry(*row)
            except (TypeError,ValueError) as error:raise ValueError('Each Q entry must be QueryEntry(pair,bucket,teacher_label)') from error
        pair=tuple(row.pair)
        if len(pair)!=2:raise ValueError('Pair identifiers must have two components')
        try:hash(pair);hash(row.bucket)
        except TypeError as error:raise ValueError('Pair and bucket identifiers must be hashable') from error
        if pair in lookup:raise ValueError('Duplicate pair in Q')
        if row.teacher_label not in (0,1):raise ValueError('Teacher labels must be binary')
        row=QueryEntry(pair,row.bucket,int(row.teacher_label));position[pair]=len(records);records.append(row);lookup[pair]=row
    return tuple(records),lookup,position

def build_control(entries,template_pairs,positive_mask,selector,*,priorities=None,
                  slot_buckets=None,slot_quotas=None,targets=None,tie_priorities=None,
                  frozen_scores=None,score_provenance=None,priority_contract='caller_defined'):
    """Reassign only negative slots, preserving fixed positives and targets.

    selector='T': cached teacher-negative pool; 'R': Q minus retained Pstar;
    'Hard': that same R pool sorted by caller FROZEN score descending, then
    caller tie priority. No ranker is fitted/called and no priorities generated.

    slot_buckets defaults to the bucket of each template pair. To impose a
    different allocation, supply an explicit target bucket for EVERY negative
    slot; slot_quotas validates this assignment rather than guessing a layout.
    Missing/infeasible priorities/capacities raise; there is no silent fallback.
    """
    if selector not in ('T','R','Hard'):raise ValueError('selector must be T, R, or Hard')
    if priority_contract not in ('common_permutation','selector_independent','caller_defined'):
        raise ValueError('Unknown priority contract')
    records,lookup,position=_entries(entries);pairs=tuple(tuple(p) for p in template_pairs)
    if len(pairs)!=len(set(pairs)):raise ValueError('Template pairs must be unique')
    if any(p not in lookup for p in pairs):raise ValueError('Every template pair, including Pstar, must belong to Q')
    mask=np.asarray(positive_mask)
    if mask.dtype!=np.dtype(bool) or mask.shape!=(len(pairs),):raise ValueError('positive_mask must be an aligned boolean vector')
    pstar={p for p,positive in zip(pairs,mask) if positive}
    if any(lookup[p].teacher_label!=1 for p in pstar):raise ValueError('Retained positives must be teacher-positive Q entries')
    if targets is None:labels=mask.astype(np.float32)
    else:
        labels=np.asarray(targets).copy()
        if labels.ndim not in (1,2) or labels.shape[0]!=len(pairs) or (labels.ndim==2 and labels.shape[1]==0):raise ValueError('Targets must have one row per fixed slot')
        primary=labels if labels.ndim==1 else labels[:,0]
        if not np.isin(primary,[0,1]).all() or not np.array_equal(primary==1,mask):raise ValueError('Primary binary targets must agree with positive_mask')
    if slot_buckets is None:buckets=tuple(lookup[p].bucket for p in pairs)
    else:
        buckets=tuple(slot_buckets)
        if len(buckets)!=len(pairs):raise ValueError('slot_buckets must align with every template slot')
        for b in buckets:
            try:hash(b)
            except TypeError as error:raise ValueError('Slot buckets must be hashable') from error
    slots=defaultdict(list)
    for j,positive in enumerate(mask):
        if not positive:slots[buckets[j]].append(j)
    quotas={b:len(v) for b,v in slots.items()}
    if slot_quotas is not None:
        declared={b:_integer(n,'quota') for b,n in slot_quotas.items()}
        if {b:n for b,n in declared.items() if n}!=quotas:
            raise ValueError('Declared quotas must exactly match negative-slot bucket assignment')
    pools={'T':defaultdict(list),'R':defaultdict(list)}
    for row in records:
        if row.pair not in pstar:
            pools['R'][row.bucket].append(row.pair)
            if row.teacher_label==0:pools['T'][row.bucket].append(row.pair)
    active=pools['T' if selector=='T' else 'R'];result=list(pairs)
    if selector=='Hard':
        if frozen_scores is None or tie_priorities is None or not isinstance(score_provenance,str) or not score_provenance.strip():
            raise ValueError('Hard requires caller frozen_scores, tie_priorities, and a nonempty score_provenance')
    elif priorities is None:raise ValueError('T/R requires explicit caller priorities')
    capacities={}
    for b,where in slots.items():
        candidates=active.get(b,[]);capacities[b]=len(candidates)
        if len(candidates)<len(where):raise ValueError(f'Insufficient {selector} capacity for bucket {b!r}: need {len(where)}, have {len(candidates)}')
        try:
            if selector=='Hard':
                values={p:float(frozen_scores[p]) for p in candidates}
                if not all(math.isfinite(v) for v in values.values()):raise ValueError('Frozen scores must be finite')
                ordered=sorted(candidates,key=lambda p:(-values[p],tie_priorities[p],position[p]))
            else:
                # Equalprioritytiesuseexplicitcaller tiekeys, then original Q
                # order. Thisworks for integer/string/arbitraryhashable IDs.
                ordered=sorted(candidates,key=lambda p:(priorities[p],tie_priorities[p] if tie_priorities is not None else position[p],position[p]))
        except KeyError as error:raise ValueError('Missing priority/score for an eligible candidate') from error
        except TypeError as error:raise ValueError('Caller priorities must be comparable within a bucket') from error
        for j,pair in zip(where,ordered):result[j]=pair
    result=tuple(result)
    assert len(set(result))==len(result)
    assert all(result[j]==pairs[j] for j in np.flatnonzero(mask))
    assert Counter(lookup[result[j]].bucket for j in np.flatnonzero(~mask))==Counter(quotas)
    assert all(result[j] not in pstar for j in np.flatnonzero(~mask))
    if selector=='T':assert all(lookup[result[j]].teacher_label==0 for j in np.flatnonzero(~mask))
    selected_teacher_positive=sum(lookup[result[j]].teacher_label==1 for j in np.flatnonzero(~mask))
    diagnostics={'selector':selector,'query_records':len(records),'fixed_positive_records':int(mask.sum()),
        'negative_budget':int((~mask).sum()),'negative_slot_quotas':quotas,'eligible_capacity_by_used_bucket':capacities,
        'changed_negative_pairs':sum(result[j]!=pairs[j] for j in np.flatnonzero(~mask)),
        'R_eligible_unretained_teacher_positive_count':sum(r.teacher_label==1 and r.pair not in pstar for r in records),
        'selected_unretained_teacher_positive_as_synthetic_negative':selected_teacher_positive,
        'priority_contract':priority_contract,'score_provenance':score_provenance if selector=='Hard' else None,
        'positives_order_targets_quotas_unchanged':True,'human_negative_truth_claim':False}
    return ControlResult(result,labels,diagnostics)
