import moments
import numarray as nx

pState = moments.MomentState()

A=nx.zeros((10,10),nx.UInt8)
A[3,1]=200
A[3,2]=10
pState.fill_moment(A)
channel = 0
for typ in ['central','spatial']:
    print typ,'moments:'
    for m in range(3):
        for n in range(3):
            try:
                moment = pState.get_moment(typ,m,n,channel)
                print '  m%d%d = % 8.2f'%(m,n,moment)
            except moments.IPPError, x:
                print '  m%d%d -> error ("%s")'%(m,n,str(x))
