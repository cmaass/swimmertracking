from threading import Thread
import copy

def test_kwargs(Nthread=4,bigfile='bigfile.txt',**kwargs):
    dumpfiles=[]
    threads=[]
    argdicts=[]
    for i in range(Nthread):
        argdicts+=[copy.deepcopy(kwargs)]
        dumpfiles+=['fish%03d.txt'%i]
        argdicts[-1]['dumpfile']=dumpfiles[-1]
        argdicts[-1]['i']=i
        
        print i, argdicts[-1]
        threads.append(Thread(target=writeFish, args=[],kwargs=argdicts[-1]))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print dumpfiles
    with open(bigfile,'wb') as newf:
        for filename in dumpfiles:
            with open(filename,'rb') as hf:
                newf.write(hf.read())     


def writeFish(**kwargs):
    d=open(kwargs['dumpfile'],'w')
    print kwargs['dumpfile']
    del kwargs['dumpfile']
    j=kwargs['i']
    del kwargs['i']
    for key in kwargs.keys():
        d.write("%s: %d\n"%(key, kwargs[key]+j))
    d.close()
        
    
test_kwargs(Nthread=5, trout=1,haddock=2,plaice=3,salmon=4)