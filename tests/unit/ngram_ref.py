#!/usr/bin/env python3
"""Port of Qwen4ExpNGramEmbedding id computation (vllm-rdna2-qwen,
amd/ple_layer.py) in plain Python, to check the C++ hasher."""
import subprocess, sys
MASK=(1<<64)-1; G=0x9E3779B97F4A7C15
def sm(v):
    v=(v+G)&MASK; v=((v^(v>>30))*0xBF58476D1CE4E5B9)&MASK; v=((v^(v>>27))*0x94D049BB133111EB)&MASK
    return (v^(v>>31))&MASK
def isp(n):
    if n<2: return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n%p==0: return n==p
    d=n-1;s=0
    while d%2==0: d//=2;s+=1
    for a in (2,325,9375,28178,450775,9780504,1795265022):
        if a%n==0: continue
        x=pow(a,d,n)
        if x in (1,n-1): continue
        for _ in range(s-1):
            x=pow(x,2,n)
            if x==n-1: break
        else: return False
    return True
V=248320; EOS=248044; NG=3; HPN=8
half=max(1,(((1<<63)-1)//V)//2)
mult=[2*(sm(1234+G*(i+1))%half)+1 for i in range(NG)]
sizes=[];offs=[];p=20000000-1;o=0
for h in range(16):
    c=p+1
    if c<=2: c=2
    else:
        if c%2==0: c+=1
        while not isp(c): c+=2
    p=c; sizes.append(p); offs.append(o); o+=p
toks=[760,6511,248044,314,9338,248044,248044,369,1,2,3,248319]
ctx=[EOS,EOS]+toks
out=[]
for col in range(2,len(ctx)):
    prev=-1
    for j in range(col):
        if ctx[j]==EOS: prev=j
    pis=col-prev-1
    sh=[]
    for s in range(NG):
        src=col-s
        sh.append(ctx[src] if (src>=0 and pis>=s) else EOS)
    ids=[]
    for n in range(2,NG+1):
        m=sh[0]*mult[0]
        for i in range(1,n): m^=sh[i]*mult[i]
        for j in range(HPN):
            h=(n-2)*HPN+j; ids.append(m%sizes[h]+offs[h])
    out.append(" ".join(map(str,ids)))
got=subprocess.run([sys.argv[1]],capture_output=True,text=True).stdout.strip().split("\n")
got=[g.strip() for g in got]
bad=[i for i,(a,b) in enumerate(zip(out,got)) if a!=b]
print("total rows", o, "| positions", len(out), "| mismatches", bad)
sys.exit(1 if bad or len(got)!=len(out) else 0)
