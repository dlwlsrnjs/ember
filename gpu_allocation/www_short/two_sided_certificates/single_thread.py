"""Explicit one-thread HiGHS execution for unchanged scientific programs.

Usage: python single_thread.py --receipt new.json path/to/program.py args...
       python single_thread.py --receipt new.json -m package.module args...

Only the documented HiGHS threads option is added. Objectives, constraints,
tolerances, seeds and time limits are preserved. Install before first solve.
"""
from pathlib import Path
import argparse,functools,hashlib,json,os,runpy,sys,time,warnings


def configure():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    import scipy.optimize as opt
    if getattr(opt.linprog,'_ndcg_highs_threads',None)==1 and getattr(opt.milp,'_ndcg_highs_threads',None)==1:return
    original_lp,original_milp=opt.linprog,opt.milp
    def options(value):
        result=dict(value or {})
        if result.get('threads',1)!=1:raise ValueError('This execution epoch requires HiGHS threads=1')
        result['threads']=1;return result
    @functools.wraps(original_lp)
    def linprog(*args,**kwargs):
        positional=list(args)
        if len(positional)>8:positional[8]=options(positional[8])
        else:kwargs['options']=options(kwargs.get('options'))
        return original_lp(*positional,**kwargs)
    @functools.wraps(original_milp)
    def milp(*args,**kwargs):
        kwargs['options']=options(kwargs.get('options'))
        return original_milp(*args,**kwargs)
    linprog._ndcg_highs_threads=milp._ndcg_highs_threads=1
    opt.linprog=linprog;opt.milp=milp
    # SciPy1.11 forwards this documented native option, warning only because
    # it is not in SciPy's short public option list. Keep other warnings.
    warnings.filterwarnings('ignore',message=r"Unrecognized options detected: \{'threads'(?:: 1)?\}\. These will be passed to HiGHS verbatim\.")
    os.environ['NDCG_HIGHS_THREADS']='1'
    os.environ['NDCG_EXECUTION_CONFIG_SHA256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('program',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.program,'Specify target program';assert not a.receipt.exists(),'Preserve earlier execution receipts'
    configure();import scipy,numpy
    meta=dict(started_unix=time.time(),pid=os.getpid(),program=a.program,highs_threads=1,
        runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scipy=scipy.__version__,numpy=numpy.__version__,
        changes='Only HiGHS threads option; problem matrices,objective,tolerance,seed,time limits unchanged',
        environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NDCG_HIGHS_THREADS')})
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(meta,indent=2)+'\n')
    try:
        if a.program[0]=='-m':
            assert len(a.program)>=2;sys.argv=a.program[1:];runpy.run_module(a.program[1],run_name='__main__',alter_sys=True)
        else:
            target=Path(a.program[0]).resolve();sys.path.insert(0,str(target.parent));sys.argv=[str(target),*a.program[1:]]
            runpy.run_path(str(target),run_name='__main__')
        meta['completed']=True
    except SystemExit as error:
        meta['completed']=error.code in (None,0);meta['exit_code']=error.code;raise
    except BaseException as error:
        meta['completed']=False;meta['exception_type']=type(error).__name__;raise
    finally:
        meta['finished_unix']=time.time();a.receipt.write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
