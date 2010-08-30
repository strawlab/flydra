function no_space(str) {
    return str.replace(" ","_");
}

function make_space(str) {
    return str.replace("_"," ");
}

function source_node_type2id(source_node_type) {
    var my_id = "selectable_" + no_space(source_node_type);
    return my_id;
}

function id2source_node_type(my_id) {
    var source_node_type = make_space(my_id.slice( "selectable_".length ));
    return source_node_type;
}

function get_select_form( source_node_type ) {
    var eid = js_client_info_global.sources[source_node_type].select_id;
    var selector = "#" + eid;
    var dom = $(selector);
    return dom;
}

function sources_selection_selected(event, ui) { 
    return sources_selection_changed(event, ui, this, "selected");
}

function sources_selection_unselected(event, ui) { 
    return sources_selection_changed(event, ui, this, "unselected");
}

// function sources_selection_stop(event, ui) { 
//     console.log("stop");
//     return sources_selection_changed(event, ui, this, "stop");
// }

function sources_selection_changed(event, ui, widget, which_change) { 
    var myid = widget.id;
    var source_node_type = id2source_node_type(myid);

    // get the original select form
    var select_form = get_select_form( source_node_type );
    var options = select_form[0].options;

    // unselect all rows in the original select form
    for (var i=0; i<options.length; i++){
	options[i].selected = 0;
    }

    // select the selected rows in the original select form
    $(".ui-selected", widget).each(function(){
    	    var index = $("#" + myid + " li").index(this);
	    var linum = $(this).attr('data-linum');
	    options[linum].selected = 1;
	});

 }

function pretty_print_row( row ) {
    //    return "pretty: " + row.id;
    return row.id;
}

function FWA_create_ul_for_items( elem ) {
	var primary;

	if (typeof js_client_info_global.dominant_source_node_type === 'undefined') {
	    for (primary in js_client_info_global.sources) {
		// set the variable "primary" to the first key
		break;
	    }
	} else {
	    primary = js_client_info_global.dominant_source_node_type;
	}


	var myitems = "";
	var linum=0;
	for (var rownum in js_client_info_global.sources[primary].rows) {
	    var row = js_client_info_global.sources[primary].rows[rownum];
	    var myid = row.id;
	    myitems = myitems + "<li class=\"ui-widget-content\" data-id=\""+row.id+"\" data-linum=\""+linum+"\">" + pretty_print_row(row) + "</li>";
	    linum++;
	    //	    debugger;
	}

	
	var my_id = source_node_type2id(primary);
	elem.append("<h2>Select data "+primary+" sources</h2>");
	//	elem.append("You've selected: <span id=\"select-result\">none</span>.");
	elem.append("<p>Hold ctrl (on Mac: command) and click to select multiple sources.</p>");
	elem.append( "<div class=\"fwa_selectable\">");
	elem.append( "<ul id=\"" + my_id + "\" class=\"ui-selectable\"> " + myitems + "</ul>" );
	elem.append( "</div>" );
	$("#"+my_id).selectable({selected: sources_selection_selected,
		unselected: sources_selection_unselected,
		    //		stop: sources_selection_stop,
		//stop: sources_selection_changed,
		    });
}

$(document).ready(function(){

	if (typeof js_client_info_global === 'undefined') {
	    return;
	}

	// With this javascript running, disable user input into all but
	// dominant selection box. This code will fill the rest.

	for (var source_node_type in js_client_info_global.sources) {
	    // dom is just the <select> box, hide its parent (the <p>)
	    // get_select_form( source_node_type ).parent().hide();
	}
	
	FWA_create_ul_for_items($("#fwa_jq_div"));
});
