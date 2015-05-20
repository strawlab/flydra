import math
import h5py

def get_valid_userblock_size( min ):
    result = 2**int(math.ceil(math.log( min, 2)))
    if result < 512:
        result = 512
    return result


def save_as_flydra_hdf5(newfilename, data, tzname, fps, smoothed_source=None,
                        smoothed_data_filename=None, raw_data_filename=None,
                        dynamic_model_name=None,
                        recording_flydra_version=None,
                        smoothing_flydra_version=None,
                        ):

    assert smoothed_source is not None
    assert smoothed_data_filename is not None
    assert raw_data_filename is not None
    assert dynamic_model_name is not None
    assert smoothing_flydra_version is not None
    assert recording_flydra_version is not None

    first_chars = '{"schema": "http://strawlab.org/schemas/flydra/1.3"}'
    pow2_bytes = get_valid_userblock_size( len(first_chars))
    userblock = first_chars + '\0'*(pow2_bytes-len(first_chars))

    with h5py.File(newfilename,'w', userblock_size=pow2_bytes) as f:
        actual_userblock_size = f.userblock_size # an AttributeError here indicates h5py is too old
        assert actual_userblock_size==len(userblock)

        for table_name, arr in data.items():
            dset = f.create_dataset( table_name, data=arr,
                                     compression='gzip',
                                     compression_opts=9)
            assert dset.compression == 'gzip'
            assert dset.compression_opts == 9
            if table_name=='trajectory_start_times':
                dset.attrs['timezone'] = tzname
                assert dset.attrs['timezone'] == tzname # ensure it is actually saved
            elif table_name=='trajectories':
                dset.attrs['frames_per_second'] = fps
                assert dset.attrs['frames_per_second'] == fps

                dset.attrs['smoothed_source'] = smoothed_source
                assert dset.attrs['smoothed_source'] == smoothed_source

                dset.attrs['smoothed_data_filename'] = smoothed_data_filename
                assert dset.attrs['smoothed_data_filename'] == smoothed_data_filename

                dset.attrs['raw_data_filename'] = raw_data_filename
                assert dset.attrs['raw_data_filename'] == raw_data_filename

                dset.attrs['dynamic_model_name'] = dynamic_model_name
                assert dset.attrs['dynamic_model_name'] == dynamic_model_name

                dset.attrs['smoothing_flydra_version'] = smoothing_flydra_version
                assert dset.attrs['smoothing_flydra_version'] == smoothing_flydra_version

                dset.attrs['recording_flydra_version'] = recording_flydra_version
                assert dset.attrs['recording_flydra_version'] == recording_flydra_version

    with open(newfilename,mode='r+') as f:
            f.write(userblock)
