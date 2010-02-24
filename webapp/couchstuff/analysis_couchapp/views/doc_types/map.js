function(doc) {
    if (doc.type=='datanode') {
        emit([doc.type,doc.properties], 1);
    } else {
        emit([doc.type,[]], 1);
    }
};
