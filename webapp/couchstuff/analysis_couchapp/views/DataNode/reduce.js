function (keys, values, rereduce) {

    array_contains = function(arr,prop) {
	for (var i=0; i< arr.length; i++) {
	    if (arr[i]==prop) return true;
	}
	return false;
    }
    // See http://www.davispj.com/2008/10/02/couchdb-joins.html
    var obj = null ;

    if (rereduce) {
        for(var i = 0 ; i < values.length ; i++) {
            if(typeof(values[i]) == 'object' && values[i].type == 'reduction') {
                if(obj == null) {
                    obj = values[i];
                } else {
                    obj.n_built   = obj.n_built   + values[i].n_built;
                    obj.n_unbuilt = obj.n_unbuilt + values[i].n_unbuilt;
                }
            }
        }
    }

    if (obj==null) {
        obj = {'type': 'reduction', // metadata for reduce
	       "n_built":0, "n_unbuilt":0};
    }

    if (keys==null) {
        return obj;
    }

    for (var i=0; i < keys.length; i++) {
        if (typeof(values[i])=='object' && values[i].type=='reduction') {
            continue;
        }

	// keys[i] = [dataset, properties, status_tags]
	status_tags = keys[i][0][2];
	if (array_contains( status_tags, "built" )) {
	    obj.n_built = obj.n_built + 1;
	} else {
	    obj.n_unbuilt = obj.n_unbuilt + 1;
	}
    }

    return obj;
}
