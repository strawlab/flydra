function(doc) {

    var is_data_node = false;
    var props = [];

    switch (doc.type) {
    case "h5":
        // adapt to DataNode
        is_data_node = true;
        if (doc.has_2d_position) {
            props = props.concat( "2d position" );
        }
        if (doc.has_2d_orientation) {
            props = props.concat( "2d orientation" );
        }
        if (doc.has_3d_position) {
            props = props.concat( "3d position" );
        }
        if (doc.has_3d_orientation) {
            props = props.concat( "3d orientation" );
        }
        if (doc.has_calibration) {
            props = props.concat( "calibration" );
        }

        var value = { properties : props,
                      start_time : doc.start_time,
                      stop_time  : doc.stop_time,
                      dataset    : doc.dataset,
                      sources    : [doc._id],
                    }
        break;
    case "datanode":
        // already implements DataNode
        value = doc;
        break;
    default:
        break;
    }

    if (is_data_node) {
        emit( [value.dataset, value.properties], value );
    }
}
