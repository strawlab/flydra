import pickle

if 1:
    fd = open('closest.pkl',mode='rb')
    by_filename = pickle.load(fd)
    fd.close()


    for filename, closest_flights_obj_ids in by_filename.iteritems():
        print filename
        print '  ',closest_flights_obj_ids
        
