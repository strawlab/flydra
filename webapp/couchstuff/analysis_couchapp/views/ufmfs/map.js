function(doc) {
    if (doc.type=="ufmf") {
        if (doc.dataset != null) {
            emit([doc.start_time,doc.stop_time], null);
        }
    }
};
