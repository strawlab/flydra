// $.couch.app() loads the design document from the server and 
// then calls our application.
$.couch.app(function(app) {
    console.log("ads 1");
    console.log(app);
    console.log("ads 2");

  // customize the couchapp profile widget with our templates and selectors
  $.extend(true, 
    app.ddoc.vendor.couchapp.evently.profile, 
    app.ddoc.evently.profile);
  
  // apply the widget to the dom
  $.log("this is app.ddoc.vendor.couchapp.evently.profile: -----------");
  $.log(app.ddoc.vendor.couchapp.evently.profile);
  $.log("-------that was app.ddoc.vendor.couchapp.evently.profile");
  $("#profile").evently(app.ddoc.vendor.couchapp.evently.profile, app);
  
  // setup the account widget
  $("#account").evently(app.ddoc.vendor.couchapp.evently.account, app);  
  
  // trigger the profile widget's events corresponding to the account widget
  $.evently.connect($("#account"), $("#profile"), ["loggedIn", "loggedOut"]);
  
  // now set up the main list of flydra_dbs
    $.log("ads 3");
  var flydra_dbs = app.ddoc.evently.flydra_dbs;
    $.log("ads 4");
  
  $.log(flydra_dbs)
    $.log("ads 4.5");
  $("#flydra_dbs").evently(flydra_dbs, app);
    $.log("ads 5");
  $.pathbinder.begin("/");
    $.log("ads 6");
});

// todo move to a plugin somewhere
// copied to toast's $.couch.app.utils
$.linkify = function(body) {
  return body.replace(/((ftp|http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-\/]))?)/gi,function(a) {
    return '<a target="_blank" href="'+a+'">'+a+'</a>';
  }).replace(/\@([\w\-]+)/g,function(user,name) {
    return '<a href="#/mentions/'+encodeURIComponent(name)+'">'+user+'</a>';
  }).replace(/\#([\w\-\.]+)/g,function(word,tag) {
    return '<a href="#/tags/'+encodeURIComponent(tag)+'">'+word+'</a>';
  });
};
