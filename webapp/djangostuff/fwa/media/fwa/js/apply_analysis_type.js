function time_get_duration_qsecs( start, stop ) {
    // get the number of "quickly computed" seconds between two times
    return dojo.date.difference(start, stop, "second");
}

function time_get_abs_qsecs( start ) {
    var reference_time = new Date(2008, 0, 0);
    reference_time.setUTCHours(0);
    return time_get_duration_qsecs( reference_time, start);
}




// -----------------------
// ID escaping: these must be kept in sync with views.py fwa_id_escape()
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
// -----------------------


function get_select_form( source_node_type ) {
    var eid = js_client_info_global.sources[source_node_type].select_id;
    var selector = "#" + eid;
    var dom = $(selector);
    return dom;
}

function get_row( rownum, source_node_type ) {
    return js_client_info_global.sources[source_node_type].rows[rownum];
}

function compute_rownums_for( source_row, target_node_type ) {
    // find the rownum in the target corresponding to the source row

    var rownums = [];
    var overlap_amounts = [];
    var targetobj = js_client_info_global.sources[target_node_type];

    for (var rownum in targetobj.rows) {
	var row = targetobj.rows[rownum];

	if (row.start_time >= source_row.start_time) {
	    if (row.stop_time <= source_row.stop_time) {
		// target row is completely within source_row time bounds
		rownums.push( rownum );
		overlap_amounts.push( 0 );
		continue;
	    }
	}

	if (source_row.start_time >= row.start_time) {
	    if (source_row.stop_time <= row.stop_time) {
		// source row is completely within target row time bounds
		rownums.push( rownum );
		overlap_amounts.push( 0 );
		continue;
	    }
	}

	// no complete overlap. compute amount of overlap.
	var last_start = Math.max( row.start_qsecs, source_row.start_qsecs );
	var first_stop = Math.min( row.stop_qsecs, source_row.stop_qsecs );

	var overlap = first_stop - last_start;
	overlap = Math.max( 0, overlap ); // clip minimum overlap to zero
	overlap_amounts.push( overlap );
    }

    if (rownums.length == 0) {
	// no perfect matches. find best overlap

	idx = 0;
	var best_overlap = 0;
	var best_idx = -1;
	for (var overlap in overlap_amounts) {
	    if (overlap > best_overlap) {
		best_overlap = overlap;
		best_idx = idx;
	    }
	    idx++;
	}
	if (best_idx != -1) {
	    rownums.push( best_idx );
	}
    }
    return rownums;
}

function sources_selection_selected(event, ui) {
    return sources_selection_changed(event, ui, this, "selected");
}

function sources_selection_unselected(event, ui) {
    return sources_selection_changed(event, ui, this, "unselected");
}

function sources_selection_changed(event, ui, widget, which_change) {
    var element = ui[which_change];

    var myid = element.parentNode.id;
    //    var myid = widget.id;
    var orig_source_node_type = id2source_node_type(myid);

    // get the original select form
    var select_form = get_select_form( orig_source_node_type );
    var options = select_form[0].options;

    var select_choices = {};
    select_choices[orig_source_node_type] = [];

    var src_rownum = $(element).attr('data-rownum');
    if (which_change=="selected") {
	select_choices[orig_source_node_type] = [src_rownum];
    }
    else { // "unselected"
	// no need to defer unselecting
	options[src_rownum].selected = 0;
    }

    var row = get_row( src_rownum, orig_source_node_type );

    var have_choices = false;
    for (var source_node_type in js_client_info_global.sources) {
	if (source_node_type == orig_source_node_type) {
	    continue;
	}
	select_choices[source_node_type] = [];

	// repeat for each of the other <select> forms, finding appropriate lines

	// get the select form
	var select_form = get_select_form( source_node_type );
	var options = select_form[0].options;
	var rownums = compute_rownums_for( row, source_node_type );
	if (which_change=="unselected") {
	    for (var optrow in rownums) {
		options[rownums[optrow]].selected = 0;
	    }
	}
	else { // "selected"
	    for (var optrow in rownums ) {
		select_choices[source_node_type].push( rownums[optrow] );
	    }
	    if (rownums.length > 1) {
		var have_choices = true;
	    }

	}
    }

    if (have_choices) {
	var prompt_str = build_impromptu_choice_dialog(select_choices);
	$.prompt(prompt_str,
		 {submit: function(v,m,f) { submit_choice_selection(v,m,f,select_choices);}, // build a callback function
			 callback: function(v,m,f) { callback_choice_selection(v,m,f,select_choices,element);}
			 });
    } else {
	if (which_change=="selected") {
	    apply_choice_selection(select_choices);
	}
    }

 }

function build_impromptu_choice_dialog(select_choices) {
	var prompt_str = '<div id="impromptu-window">\n';
	for (var source_node_type in select_choices) {
	    var this_choices = select_choices[source_node_type];
	    if (this_choices.length>1)  {
		var radio_group_name = source_node_type; // XXX escape?
		var ids = [];
		var values = [];
		var texts = [];
		for (var i=0; i<this_choices.length; i++){
		    var rownum = this_choices[i];
		    ids.push( rownum ); // XXX escape?
		    values.push( rownum ); // line numbers

		    var row = get_row( rownum, source_node_type);
		    var mytext = pretty_print_row(row) + ' (id="'+row.id+'", status_tags="'+row.status_tags+'")';
		    texts.push( mytext );
		}

		prompt_str = prompt_str + '<label for="'+ids[0]+'">Which '+source_node_type+' node do you want to use?</label><ul>';
		for (var i=0; i<this_choices.length; i++){
		    var checked_str='';
		    if (i==0) {
			checked_str = ' checked="checked"';
		    }
		    prompt_str = prompt_str + '<li><input type="radio" name="'+radio_group_name+'" id="'+ids[i]+'" value="'+values[i]+'"'+checked_str+'/>' + texts[i]+'</li>';
		}
		prompt_str  = prompt_str + '</ul>';
	    }
	}
	prompt_str  = prompt_str +      '</div>\n';
    return prompt_str;
}

function submit_choice_selection( v,m,f, select_choices ) {
    // the submit callback is called only when "OK is pressed"

    // apply the user feedback to get an updated result
    for (var node_type in f) {
	select_choices[node_type] = [f[node_type]];
    }

    // check whether the result is good
    var is_bad = false;
    for (var node_type in select_choices) {
	if (select_choices[node_type].length != 1) {
	    is_bad = true;
	}
    }

    if (is_bad) {
	// error parsing data, user prompted again
	return false;
    }
    apply_choice_selection(select_choices);
    return true;
}

function callback_choice_selection( v,m,f, select_choices, element ) {

    // The "callback" callback is called whether the user pressed OK
    // or cancelled.

    // Unless we were successful, we need to de-select the original
    // element to keep in sync with select boxes.

    // check whether the result is good
    var is_bad = false;
    for (var node_type in select_choices) {
	if (select_choices[node_type].length != 1) {
	    is_bad = true;
	}
    }

    if (is_bad) {
	//unselect the original selectable
	$(element).removeClass("ui-selected");
    }
}

function apply_choice_selection( select_choices ) {
    for (var source_node_type in select_choices ) {
	// get the original select form
	var select_form = get_select_form( source_node_type );
	var options = select_form[0].options;

	var idx = select_choices[source_node_type][0];
	options[idx].selected = 1;
    }
}

function pretty_print_row( row ) {
    var fmt = 'HH:mm, d MMM, yyyy';

    var duration_seconds = dojo.date.difference(row.start_time_obj, row.stop_time_obj, "second");
    var duration_minutes = dojo.date.difference(row.start_time_obj, row.stop_time_obj, "minute");
    var duration_hours = dojo.date.difference(row.start_time_obj, row.stop_time_obj, "hour");
    var duration_days = dojo.date.difference(row.start_time_obj, row.stop_time_obj, "day");
    var str1 = dojo.date.locale.format(row.start_time_obj, {datePattern: "d MMM yyyy"});

    //str1 =  " ("+row.duration_qsecs+" seconds) at " + str1;
    str1 =  " at " + str1;
    if (duration_days) {
	return duration_days + " days" + str1;
    }
    else if (duration_hours) {
	return duration_hours + " hours" + str1;
    }
    else if (duration_minutes) {
	return duration_minutes + " minutes" + str1;
    }
    else if (duration_seconds) {
	return duration_seconds + " seconds" + str1;
    }
    return "an unknown duration" + str1;
}

function sort_by_start_time(a, b) {
    var x = a.start_time_obj;
    var y = b.start_time_obj;
    return ((x < y) ? -1 : ((x > y) ? 1 : 0));
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

	// fill global object with parsed dates so we don't do it later
	for (var source_node_type in js_client_info_global.sources) {
	    var mysource = js_client_info_global.sources[source_node_type];
	    for (var rownum in mysource.rows) {
		var row = mysource.rows[rownum];
		var start = dojo.date.stamp.fromISOString(row.start_time);
		var stop = dojo.date.stamp.fromISOString(row.stop_time);

		var start_qsecs = time_get_abs_qsecs( start );
		var duration_qsecs = time_get_duration_qsecs( start, stop );

		mysource.rows[rownum].start_time_obj = start;
		mysource.rows[rownum].stop_time_obj = stop;

		mysource.rows[rownum].start_qsecs = start_qsecs;
		mysource.rows[rownum].duration_qsecs = duration_qsecs;
		mysource.rows[rownum].stop_qsecs = time_get_abs_qsecs( stop );
	    }
	}

	// Sort rows by start time
	js_client_info_global.sources[primary].rows.sort(sort_by_start_time);

	var myitems = "";
	for (var rownum in js_client_info_global.sources[primary].rows) {
	    var row = js_client_info_global.sources[primary].rows[rownum];
	    var myid = row.id;
	    myitems = myitems + "<li class=\"ui-widget-content\" data-rownum=\""+rownum+"\" data-id=\""+row.id+"\">" + pretty_print_row(row) + "</li>";
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
		    });
}

function true_init() {

	if (typeof js_client_info_global === 'undefined') {
	    return;
	}

	// With this javascript running, disable user input into all but
	// dominant selection box. This code will fill the rest.

	for (var source_node_type in js_client_info_global.sources) {
	    // dom is just the <select> box, hide its parent (the <p>)

	    var select_form = get_select_form( source_node_type );
	    var options = select_form[0].options;
	    // unselect all rows in the original select form
	    for (var i=0; i<options.length; i++){
		options[i].selected = 0;
	    }

	    get_select_form( source_node_type ).parent().hide();
	}
	
	FWA_create_ul_for_items($("#fwa_jq_div"));
}

$(document).ready(function(){
	// jQuery is loaded. Now load dojo (for date processing).
	dojo.require("dojo.date");
	dojo.require("dojo.date.locale");
	dojo.require("dojo.date.stamp");
	dojo.addOnLoad(true_init);
});
