import geom
import reconstruct

count = 0
for x1 in [1,100,10000]:
    for y1 in [5,50,500]:
        for z1 in [-10,234,0]:
            for x2 in [3,50]:
                pt_a = [x1,y1,z1,1]
                pt_b = [x2,5,6,1]
                hz_p = reconstruct.pluecker_from_verts(pt_a,pt_b)

                a=geom.ThreeTuple(pt_a[:3])
                b=geom.ThreeTuple(pt_b[:3])
                L = geom.line_from_points(a,b)

                hzL = geom.line_from_HZline(hz_p)

                strL = str(L)
                strhzL = str(hzL)
                assert strL==strhzL
                count += 1
                if 0:
                    print 'hz_p',hz_p
                    print 'correct',L
                    print 'test   ',hzL
                    print
print '%d tests passed'%count

line=geom.line_from_points(geom.ThreeTuple((2,1,0)),
                           geom.ThreeTuple((2,0,0)))
print line
print line.closest()
print line.dist2()
                      
