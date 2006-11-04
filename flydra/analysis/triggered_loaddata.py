f_segments = [line.strip().split() for line in analysis_file.readlines() if not line.strip().startswith('#')]

try:
    # maybe pre-loaded if we're in ipython
    h5files
except NameError:
    # load
    h5files = {}
    parsed = {}

    heading_early = {}
    heading_late = {}
    turn_angle = {}
    early_xvel = {}
    early_yvel = {}

    late_xvel = {}
    late_yvel = {}

    xs_pre = {}
    ys_pre = {}
    xs_post= {}
    ys_post= {}

    trig_fnos = {}
    h5filenames = {}

    print 'loading data...'
    for line in f_segments:
        upwind, fstart, trig_fno, fend, h5filename, condition_float = line
        upwind = bool(upwind)
        fstart = int(fstart)
        trig_fno = int(trig_fno)
        fend = int(fend)
        condition_float = float(condition_float)
        if h5filename not in h5files:
            h5files[h5filename] = result_browser.get_results(h5filename)
            f,xyz,L,err,ts = result_browser.get_f_xyz_L_err( h5files[h5filename],include_timestamps=True )
            parsed[h5filename] = f,xyz,L,err,ts

        results = h5files[h5filename]
        f,xyz,L,err,timestamps = parsed[h5filename]

        early_start = trig_fno-10
        early_end = trig_fno

        late_start = trig_fno+15
        late_end = trig_fno+25

        early_cond = (early_start <= f) & (f<=early_end)
        late_cond = (late_start <= f) & (f<=late_end)

        latency = 5 # n frames
        pre_idx = (fstart <= f) & (f<=(trig_fno+latency))
        post_idx = ((trig_fno+latency) <= f) & (f<=fend)
        pre_xyz = xyz[pre_idx]
        post_xyz = xyz[post_idx]

        early_times = timestamps[early_cond]
        late_times = timestamps[late_cond]

        early_dur = early_times[-1]-early_times[0]
        late_dur = late_times[-1]-late_times[0]

        early_xyz = xyz[early_cond]
        late_xyz = xyz[late_cond]

        xyz_dist_early = (early_xyz[-1]-early_xyz[0])/1000.0
        xyz_dist_late = (late_xyz[-1]-late_xyz[0])/1000.0

        heading_early.setdefault(condition_float,[]).append( math.atan2( xyz_dist_early[1], xyz_dist_early[0] ) )
        heading_late.setdefault(condition_float,[]).append( math.atan2( xyz_dist_late[1], xyz_dist_late[0] ) )

        turn_angle.setdefault(condition_float,[]).append(  (heading_late[condition_float][-1] -
                                                  heading_early[condition_float][-1])%(2*pi) )

        scalar_early_xvel = float( xyz_dist_early[0]/early_dur )
        early_xvel.setdefault(condition_float,[]).append( scalar_early_xvel )
        early_yvel.setdefault(condition_float,[]).append( xyz_dist_early[1]/early_dur )
        late_xvel.setdefault(condition_float,[]).append( xyz_dist_late[0]/late_dur )
        late_yvel.setdefault(condition_float,[]).append( xyz_dist_late[1]/late_dur )

        xs_pre.setdefault(condition_float,[]).append( pre_xyz[:,0] )
        ys_pre.setdefault(condition_float,[]).append( pre_xyz[:,1] )    
        xs_post.setdefault(condition_float,[]).append( post_xyz[:,0] )
        ys_post.setdefault(condition_float,[]).append( post_xyz[:,1] )    

        trig_fnos.setdefault(condition_float,[]).append( trig_fno )
        h5filenames.setdefault(condition_float,[]).append( h5filename )
    print 'done'

    # convert data to numpy
    for d in [heading_early,
              heading_late,
              turn_angle,
              early_xvel,
              early_yvel,
              late_xvel,
              late_yvel,
              xs_pre,
              ys_pre,
              xs_post,
              ys_post]:
        for condition_float in d.keys():
            d[condition_float] = nx.asarray(d[condition_float])

