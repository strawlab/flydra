import result_browser
import glob
import sets
import FlyMovieFormat

def main():
    small_fmfs = glob.glob('small_*smd')
    basenames = [ '_'.join(fname.split('_')[:3])+'_%s' for fname in small_fmfs ]
    basenames = list(sets.Set(basenames)) # find unique only
    print basenames
    h5files = glob.glob('*.h5')
    print h5files
    for fname in h5files:
        results = result_browser.get_results(fname,mode='r+')
        result_browser.simple_add_backgrounds(results)
        results.close()

        for basename in basenames:
            print '-='*20
            print
            print fname, basename
            print
            print '-='*20
            
            results = result_browser.get_results(fname,mode='r+')
            try:
                for cam_id in ['cam1_0',
                               'cam2_0',
                               'cam3_0',
                               'cam4_0',
                               'cam5_0',
                               ]:
                    #fname = 'small_20060515_190908_%s'%cam_id
                    fmf_fname = basename%cam_id
                    #result_browser.update_exact_roi_movie_info(results,cam_id,fname)
                    try:
                        result_browser.update_small_fmf_summary(results,cam_id,fmf_fname)
                    except FlyMovieFormat.InvalidMovieFileException:
                        print 'skipping broken .fmf file: %s'%fmf_fname
                        pass
            finally:
                results.close()
        
if 0:
    import hotshot
    prof = hotshot.Profile("profile.hotshot")
    try:
        res = prof.runcall(main)
    finally:
        prof.close()
else:
    main()
