function(doc) {

    var is_data_node = false;
    var props = [];

    switch (doc.type) {
    case "h5":
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
        break;
    case "ufmf":
        is_data_node = true;
        props = props.concat( "ufmf" );
        break;
    default:
        break;
    }

    if (is_data_node) {
        if (typeof(doc.start_time)=="undefined") {
            is_data_node = false;
        }
    }

    if (is_data_node) {
        var value = { properties : props,
                      start_time : doc.start_time,
                      stop_time  : doc.stop_time,
                      dataset    : doc.dataset,
                      sources    : [doc._id],
                    }
        emit( null, value );
    }
}
