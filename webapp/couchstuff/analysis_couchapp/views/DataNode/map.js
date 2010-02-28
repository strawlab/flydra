function(doc) {

    var success = true;
    var props = [];
    var value;

    switch (doc.type) {
    case "h5":
        // adapt to DataNode
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

        value = { properties : props,
                  start_time : doc.start_time,
                  stop_time  : doc.stop_time,
                  dataset    : doc.dataset,
                  sources    : [doc._id],
		  status_tags: ["built"],
                }
        break;
    case "calibration":
        value = { properties : ["calibration"],
                  dataset    : doc.dataset,
                  sources    : [doc._id],
		  status_tags: ["built"],
                }
        break;
    case "datanode":
        // already implements DataNode
        value = doc;
        break;
    default:
        success = false;
        break;
    }

    if (success) {
        emit( [value.dataset, value.properties, value.status_tags], value );
    }
}
