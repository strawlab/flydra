// $.couch.app() loads the design document from the server and 
// then calls our application.
$.couch.app(function(app) {
    //console.log(app);

  // customize the couchapp profile widget with our templates and selectors
  $.extend(true, 
    app.ddoc.vendor.couchapp.evently.profile, 
    app.ddoc.evently.profile);
  
  // apply the widget to the dom
  $("#profile").evently(app.ddoc.vendor.couchapp.evently.profile, app);
  
  // setup the account widget
  $("#account").evently(app.ddoc.vendor.couchapp.evently.account, app);  
  
  // trigger the profile widget's events corresponding to the account widget
  $.evently.connect($("#account"), $("#profile"), ["loggedIn", "loggedOut"]);
  
  $.pathbinder.begin("/");
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
