function(doc) {
    if (doc.type=="ufmf") {
        emit([doc.start_time,doc.stop_time,doc.cam_id], doc);
    }
};
