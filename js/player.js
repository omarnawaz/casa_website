/* One clock for two films ("Running the question backwards" on the Research page).
   A single Play button and a single scrubber drive both simulations, frame for frame, and the month
   they have reached is printed beside the scrubber.
   The films never start by themselves. Without this file, or if a film cannot be played this way,
   each <video> keeps the browser's own controls and the shared ones stay hidden.

   The markup is made by film_pair in templates/research.html; the number of frames, the frames shown
   each second and the year the films cover come from data/research.yaml. */
(function () {
  'use strict';

  var root = document.querySelector('[data-pair]');
  if (!root) return;

  var videos = Array.prototype.slice.call(root.querySelectorAll('video'));
  var lead = videos[0], follower = videos[1];            // the first film is the clock; the second follows it
  var controls = root.querySelector('.research-pair__controls');
  var button = controls && controls.querySelector('button');
  var scrub = controls && controls.querySelector('input');
  var track = controls && controls.querySelector('.research-pair__track');
  var label = controls && controls.querySelector('.research-pair__month');
  if (!lead || !follower || !button || !scrub || !track) return;

  var FPS = Number(root.getAttribute('data-fps')) || 10;
  var YEAR = root.getAttribute('data-year') || '';
  var LAST = Number(scrub.max);                          // the last frame
  var DURATION = (LAST + 1) / FPS;                       // in seconds
  var HELD = LAST + 1 - 365;                             // the films hold 1 January for their first few frames, then show one day in each frame
  var DAILY = HELD >= 0 && HELD < 31;                    // a film of any other length has the year spread evenly over it
  var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  var MONTH_ENDS = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];
  var WEEK = Math.round((LAST + 1) * 7 / 365);           // a week, in frames

  var time = Number(scrub.value) / FPS;                  // start on the frame the posters show
  var linked = false, playing = false, primed = false, raf = 0, shownAt = -1, shownMonth = '';

  function monthAt(frame) {
    var day = DAILY ? Math.floor(frame + 0.001) - HELD : Math.floor(frame / (LAST + 1) * 365), m = 0;
    day = Math.max(0, Math.min(364, day));
    while (day >= MONTH_ENDS[m]) m++;
    return (MONTHS[m] + ' ' + YEAR).trim();
  }

  function render() {                                    // runs on every frame, so it touches the page only on a change
    var at = Math.max(0, Math.min(LAST, time * FPS));    // not rounded to a frame, so the dot glides while the films step
    if (at !== shownAt) {
      shownAt = at;
      scrub.value = at;
      track.style.setProperty('--p', (at / LAST).toFixed(4));
    }
    var month = monthAt(at);
    if (month !== shownMonth) {
      shownMonth = month;
      scrub.setAttribute('aria-valuetext', month);       // read out by a screen reader
      if (label) label.textContent = month;              // and printed beside the scrubber
    }
  }

  function cue(v) { if (time > 0) v.currentTime = time; }   // to the frame the posters show, or wherever the scrubber has been put

  // The films are not fetched until someone presses Play or takes hold of the scrubber, so a visitor who
  // scrolls past does not pay for them. Until then the posters show.
  function prime() {
    if (primed) return;
    primed = true;
    videos.forEach(function (v) { v.preload = 'auto'; if (v.readyState > 0) cue(v); else v.load(); });
  }

  function seek(t) {
    time = Math.max(0, Math.min(DURATION - 0.05, t));
    videos.forEach(function (v) { if (v.readyState > 0) v.currentTime = time; });
    render();
  }

  function setPlaying(on) {
    playing = on;
    root.classList.toggle('is-playing', on);
    button.textContent = on ? 'Pause' : 'Play';
    button.setAttribute('aria-label', (on ? 'Pause' : 'Play') + ' both simulations');
  }

  function tick() {
    cancelAnimationFrame(raf);
    time = lead.currentTime;
    if (!follower.seeking && follower.readyState > 1 && Math.abs(follower.currentTime - time) > 0.2) follower.currentTime = time;
    render();
    if (playing) raf = requestAnimationFrame(tick);
  }

  function pause() {
    cancelAnimationFrame(raf);
    videos.forEach(function (v) { v.pause(); });
    setPlaying(false);
    if (lead.readyState > 0) seek(lead.currentTime);     // leave both films on the same frame
  }

  function play() {
    prime();
    if (time >= DURATION - 0.2) seek(0);
    setPlaying(true);
    Promise.all(videos.map(function (v) { return v.play(); })).then(function () {
      if (!playing) return;
      if (Math.abs(lead.currentTime - time) > 1) lead.currentTime = time;   // in case the cue at loadedmetadata did not take
      follower.currentTime = lead.currentTime;
      tick();
    }).catch(function (error) {
      if (error && error.name === 'NotSupportedError') unlink(); else pause();
    });
  }

  function toggle() { if (!linked) return; if (playing) pause(); else play(); }

  function link(on) {                                    // on: one shared clock; off: each film on the browser's own controls
    linked = on;
    videos.forEach(function (v) { v.controls = !on; v.loop = on; });
    controls.hidden = !on;
    root.classList.toggle('is-linked', on);
  }

  function unlink() { if (linked) { pause(); link(false); } }

  videos.forEach(function (v) {
    v.addEventListener('loadedmetadata', function () { if (linked && primed) cue(v); });
    v.addEventListener('click', toggle);
  });
  root.addEventListener('error', unlink, true);          // a film that fails to load or to decode (these errors do not bubble)
  button.addEventListener('click', toggle);
  scrub.addEventListener('input', function () { prime(); seek(Number(scrub.value) / FPS); });
  scrub.addEventListener('pointerdown', prime);
  scrub.addEventListener('keydown', function (event) {   // the arrow keys move a week at a time, not one frame; the space bar plays and pauses
    if (event.key === ' ') { event.preventDefault(); toggle(); return; }
    var step = { ArrowLeft: -WEEK, ArrowDown: -WEEK, ArrowRight: WEEK, ArrowUp: WEEK }[event.key];
    if (!step) return;
    event.preventDefault();
    prime();
    seek((Math.round(Number(scrub.value)) + step) / FPS);
  });

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {        // stop the films once the figure has gone from view
      entries.forEach(function (entry) { if (!entry.isIntersecting && playing) pause(); });
    }, { rootMargin: '300px 0px' }).observe(root);
  }

  link(true);
  render();
})();
