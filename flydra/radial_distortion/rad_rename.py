import glob
files = glob.glob('mama*.rad')
files.sort()
for i,fname in enumerate(files):
    print 'ln %s %s'%(fname,'basename%d.rad'%(i+1,))
