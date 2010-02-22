function (keys, values, rereduce) {
    // See http://www.davispj.com/2008/10/02/couchdb-joins.html
    var obj = null ;

    if (rereduce) {
        for(var i = 0 ; i < values.length ; i++) {
            if(typeof(values[i]) == 'object' && values[i].type == 'reduction') {
                if(obj == null) {
                    obj = values[i];
                } else {
                    if(obj.dataset_doc == null) {
                        obj.dataset_doc = values[i].dataset_doc;
                    }
                    obj.ufmf_bytes = obj.ufmf_bytes + values[i].ufmf_bytes;
                    obj.h5_bytes = obj.h5_bytes + values[i].h5_bytes;
                }
            }
        }
    }

    if (obj==null) {
        obj = {"ufmf_bytes":0, "h5_bytes":0, 'type': 'reduction', "dataset_doc":null};
    }

    if (keys==null) {
        return obj;
    }

    for (var i=0; i < keys.length; i++) {
        if (typeof(values[i])=='object' && values[i].type=='reduction') {
            continue;
        }

        if (keys[i][0][1]==0) {
            // doc.type=="dataset", value = doc.name
            obj.dataset_doc = values[i];
        } else if (keys[i][0][1]==1) {
            // doc.type=="ufmf", value = doc.filesize
            obj.ufmf_bytes = obj.ufmf_bytes + values[i];
        } else if (keys[i][0][1]==2) {
            // doc.type=="h5", value = doc.filesize
            obj.h5_bytes = obj.h5_bytes + values[i];
        }
    }
    return obj;
}
