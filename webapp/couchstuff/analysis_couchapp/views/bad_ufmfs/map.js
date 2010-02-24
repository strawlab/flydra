function(doc) {
    if (doc.type=="ufmf") {
        if (doc.dataset != null) {
            if (doc.start_time == "xxx") {
                emit([doc.start_time,doc.stop_time,doc.cam_id], null);
            }
        }
    }
};
