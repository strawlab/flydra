import os
import drive_cum_analysis
import numpy as np
import scipy.io


source='MLE_position'
#source='kalman'

if 1:

    bad_col_names = ['xvel', 'yvel', 'zvel', 'P00', 'P01', 'P02',
                     'P11', 'P12', 'P22', 'P33', 'P44',
                     'P55','orig_data_present','timestamp']

    flyids = drive_cum_analysis.no_post_experiments
    flyids.sort()
    all_arrs = []
    csv = []
    for flyid_enum,flyid in enumerate(flyids):
        flyid_matlab = flyid_enum+1
        csv.append( (str(flyid_matlab),
                     os.path.splitext(flyid._kalman_filename)[0]) )
        list_of_rows=flyid.get_list_of_kalman_rows_by_source(source=source,
                                                             flystate='flying')
        if source=='MLE_position':
            # filter bad obj_ids
            good_obj_ids = []
            list_of_good_obj_id_rows=flyid.get_list_of_kalman_rows_by_source(
                source='kalman',flystate='flying')
            for good_kalman_rows in list_of_good_obj_id_rows:
                good_obj_id = np.unique(good_kalman_rows['obj_id'])
                assert len(good_obj_id)==1
                good_obj_ids.append( good_obj_id[0] )

        for kalman_rows in list_of_rows:
            if source=='MLE_position':
                # filter bad obj_ids
                obj_id = np.unique(kalman_rows['obj_id'])
                assert len(obj_id)==1
                obj_id = obj_id[0]
                if obj_id not in good_obj_ids:
                    continue

            flyid_arr = np.empty( (len(kalman_rows),), dtype=np.float )
            flyid_arr.fill(flyid_matlab)
            orig_cols = kalman_rows.dtype.names
            new_arrs = [flyid_arr]
            new_names = ['file_id']
            for orig_col in orig_cols:
                if orig_col in bad_col_names:
                    continue
                new_arrs.append( kalman_rows[orig_col] )
                new_names.append( orig_col )
            newra = np.rec.fromarrays( new_arrs, names=new_names )
            all_arrs.append(newra)
    all_arrs = np.hstack(all_arrs)

    s=dict(zip(all_arrs.dtype.names,
               [all_arrs[n] for n in all_arrs.dtype.names]))

    cond_name = 'nopost'
    scipy.io.savemat('%s-%s.mat'%(cond_name,source),s)
    fd = open('%s-%s.csv'%(cond_name,source),mode='wb')
    for row in csv:
        fd.write( ','.join(row) + '\n' )
    fd.close()
