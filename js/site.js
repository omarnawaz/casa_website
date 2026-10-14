/* CASA: what every page shares.
   1. Tiles that open: only one answer in a group is open at a time.
   2. A few dots of "air" in the hero. They drift for a few seconds as the page arrives,
      settle, and stir again whenever the pointer moves through them.
   3. Films: a Play button and a thin bar that shows how far the film has run.
   4. Figures that open larger.
   The atlas on the home page is js/atlas.js; the cards on the Group page are js/cards.js. */
(function () {
  'use strict';

  var CASA = window.CASA = window.CASA || {};
  var lessMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  function each(nodes, fn) { Array.prototype.forEach.call(nodes, fn); }
  function listen(query, fn) {
    if (query.addEventListener) query.addEventListener('change', fn);
    else if (query.addListener) query.addListener(fn);
  }


  /* ---------- 1. one answer at a time ---------- */

  // Browsers that know <details name> do this themselves; this covers the rest.
  var asks = document.querySelectorAll('details[name]');
  each(asks, function (ask) {
    ask.addEventListener('toggle', function () {
      if (!ask.open) return;
      each(asks, function (other) { if (other !== ask && other.name === ask.name) other.open = false; });
    });
  });


  /* ---------- 2. dots of air ---------- */

  (function () {
    var canvas = document.querySelector('.hero__air');
    var hero = canvas && canvas.parentElement;
    var ctx = canvas && canvas.getContext && canvas.getContext('2d');
    if (!ctx) return;

    // The dots take their colours from the hero (see the .hero-… classes in site.css):
    // two bright ones, and the colour of the lettering.
    var look = window.getComputedStyle(hero);
    function token(name, fallback) { return (look.getPropertyValue(name) || '').trim() || fallback; }
    var LETTERS = token('--hero-fg', '#fff5e3'), ONE = token('--air-1', '#ffc72e'), TWO = token('--air-2', '#ff9ec6');
    var COLOURS = [ONE, TWO, LETTERS, TWO, ONE];
    var REACH = 150;           // how close the pointer has to come, in px
    var FIRST_BREATH = 1800;   // the air moves by itself only as the page arrives: this long, then it dies down
    var STIR = 4000;           // how long it keeps moving after the pointer last stirred it, in ms
    var dots = [];
    var words = [];            // boxes around the lettering, with a wide margin: no dot shows inside one
    var blocks = [];           // boxes around a picture or the people: a dot fades out as it passes behind one
    var width = 0, height = 0, ratio = 1;
    var pointer = { x: 0, y: 0, on: false };
    var frame = 0, last = 0;
    var inView = true;
    var wind = 0;              // 0 = still, 1 = drifting
    var awakeUntil = 0;

    // A small fixed-seed generator, so the dots start in the same places on every visit.
    var seed = 20;
    function rand() {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    }

    function makeDots() {
      var count = Math.max(7, Math.min(12, Math.round(width * height / 105000)));   // a few, not a snowstorm
      dots = [];
      for (var i = 0; i < count; i++) {
        var colour = COLOURS[i % COLOURS.length];
        var size = rand() * rand();             // mostly small, a few larger
        dots.push({
          x: rand(), y: rand(),                 // position as a share of the field
          // Dots the colour of the lettering stay small, so they never read as a blemish on it.
          r: colour === LETTERS ? 3 + size * 2 : 4 + size * 6,
          vx: 5 + rand() * 13,                  // a slow wind from the left, px per second
          vy: -2.5 + rand() * 5,
          px: 0, py: 0,                         // push from the pointer
          a: 1,                                 // how solid it is: 0 while it is over words or behind a block
          phase: rand() * 6.283,
          colour: colour
        });
      }
    }

    // Everything in the hero says what it is to the air with a data-air attribute:
    // "lines" for large type (measured line by line, so a dot may sit beside a short line),
    // "block" for a paragraph or a row of buttons, "solid" for a picture or a person.
    function findWords() {
      var field = hero.getBoundingClientRect();
      words = [];
      function keep(box, sideways, updown) {
        // A dot too close to a line of words reads as stray punctuation, so each box has a wide margin.
        words.push({
          left: box.left - field.left - sideways, right: box.right - field.left + sideways,
          top: box.top - field.top - updown, bottom: box.bottom - field.top + updown
        });
      }

      // The strip the navigation sits in stays clear from edge to edge.
      var nav = document.querySelector('.top__in');
      if (nav) words.push({ left: -40, right: field.width + 40, top: -40, bottom: nav.getBoundingClientRect().bottom - field.top + 4 });

      each(hero.querySelectorAll('[data-air="lines"]'), function (el) {
        var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
        var node, range = document.createRange();
        while ((node = walker.nextNode())) {
          if (!node.nodeValue.trim()) continue;
          range.selectNodeContents(node);
          each(range.getClientRects(), function (box) { if (box.width > 2) keep(box, 34, 2); });
        }
      });
      each(hero.querySelectorAll('[data-air="block"]'), function (el) { keep(el.getBoundingClientRect(), 30, 22); });
    }

    function findBlocks() {
      var field = hero.getBoundingClientRect();
      blocks = Array.prototype.map.call(hero.querySelectorAll('[data-air="solid"]'), function (el) {
        var box = el.getBoundingClientRect();
        return {
          left: box.left - field.left - 22, right: box.right - field.left + 22,
          top: box.top - field.top - 22, bottom: box.bottom - field.top + 22
        };
      });
    }

    function inBoxes(boxes, x, y) {
      for (var i = 0; i < boxes.length; i++) {
        var b = boxes[i];
        if (x > b.left && x < b.right && y > b.top && y < b.bottom) return b;
      }
      return null;
    }
    function hidden(x, y) { return inBoxes(words, x, y) || inBoxes(blocks, x, y); }

    function measure() {
      ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = hero.clientWidth;
      height = hero.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      findWords();
      findBlocks();
      if (!dots.length) makeDots();
      // Nothing starts on top of the words, half hidden behind a block, cut off by an edge, or crowding another dot.
      function crowded(d, n) {
        for (var k = 0; k < n; k++) {
          var dx = (d.x - dots[k].x) * width, dy = (d.y - dots[k].y) * height;
          if (dx * dx + dy * dy < 3600) return true;
        }
        return false;
      }
      function onEdge(d) {
        var x = d.x * width, y = d.y * height;
        return x < 18 || x > width - 18 || y < 18 || y > height - 18;
      }
      dots.forEach(function (d, n) {
        for (var tries = 0; tries < 120 && (onEdge(d) || hidden(d.x * width, d.y * height) || crowded(d, n)); tries++) {
          d.x = rand();
          d.y = rand();
        }
        // On a small screen there may be no free place left: such a dot stays out of sight until it drifts clear.
        d.a = hidden(d.x * width, d.y * height) ? 0 : 1;
      });
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      for (var i = 0; i < dots.length; i++) {
        var d = dots[i];
        if (d.a <= 0) continue;
        ctx.globalAlpha = d.a;
        ctx.fillStyle = d.colour;
        ctx.beginPath();
        ctx.arc(d.x * width, d.y * height, d.r, 0, 6.2832);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function step(now) {
      frame = 0;
      var dt = Math.min(0.05, (now - last) / 1000 || 0);
      last = now;
      var t = now / 1000;
      var calm = Math.pow(0.06, dt);            // a push fades within about a second
      var moving = 0;

      // The wind picks up quickly and dies down over about three seconds.
      var target = now < awakeUntil ? 1 : 0;
      wind += (target - wind) * Math.min(1, dt * (target ? 2.5 : 1.5));

      for (var i = 0; i < dots.length; i++) {
        var d = dots[i];
        var x = d.x * width, y = d.y * height;
        if (pointer.on) {
          var dx = x - pointer.x, dy = y - pointer.y;
          var dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (dist < REACH) {
            var f = 1 - dist / REACH;
            d.px += dx / dist * f * f * 2600 * dt;
            d.py += dy / dist * f * f * 2600 * dt;
          }
        }
        // Two dots never sit on top of one another.
        for (var j = i + 1; j < dots.length; j++) {
          var e = dots[j];
          var ex = x - e.x * width, ey = y - e.y * height;
          var gap = Math.sqrt(ex * ex + ey * ey) || 1;
          var room = d.r + e.r + 10;
          if (gap < room) {
            var shove = (1 - gap / room) * 900 * dt;
            d.px += ex / gap * shove; d.py += ey / gap * shove;
            e.px -= ex / gap * shove; e.py -= ey / gap * shove;
          }
        }
        // No dot comes to rest cut in half by an edge: at the sides it is carried on with the wind
        // (off one side and in at the other), and at the bottom it is lifted back into the field.
        if (x < 18 || x > width - 18) d.px += 420 * dt;
        if (y > height - 18) d.py -= 420 * dt;
        x += (d.vx * wind + d.px) * dt;
        y += ((d.vy + Math.sin(t * 0.5 + d.phase) * 5) * wind + d.py) * dt;
        d.px *= calm;
        d.py *= calm;
        moving += Math.abs(d.px) + Math.abs(d.py);
        if (x > width + 14) x = -14; else if (x < -14) x = width + 14;
        if (y > height + 14) y = -14; else if (y < -14) y = height + 14;
        // A dot never shows lying across a word or half hidden behind a block: it fades away as it gets there and back as it leaves.
        var goal = hidden(x, y) ? 0 : 1;
        if (d.a !== goal) {
          d.a += (goal - d.a) * Math.min(1, dt * 10);
          if (Math.abs(goal - d.a) < 0.03) d.a = goal; else moving += 2;
        }
        d.x = x / width;
        d.y = y / height;
      }
      draw();
      // Once everything has settled, stop drawing until the air is stirred again.
      if (target || wind > 0.01 || moving > 1) play();
    }

    function play() {
      if (frame || lessMotion.matches || !inView || document.hidden) return;
      frame = window.requestAnimationFrame(step);
    }
    function pause() {
      if (frame) window.cancelAnimationFrame(frame);
      frame = 0;
      last = 0;
    }
    function stir(ms) {
      awakeUntil = Math.max(awakeUntil, performance.now() + ms);
      play();
    }

    hero.addEventListener('pointermove', function (e) {
      if (e.pointerType === 'touch') return;
      var box = hero.getBoundingClientRect();
      pointer.x = e.clientX - box.left;
      pointer.y = e.clientY - box.top;
      pointer.on = true;
      stir(STIR);
    });
    hero.addEventListener('pointerleave', function () { pointer.on = false; });

    if ('ResizeObserver' in window) new ResizeObserver(measure).observe(hero);
    else window.addEventListener('resize', measure);

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        inView = entries[0].isIntersecting;
        if (inView) play(); else pause();
      }).observe(hero);
    }
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) pause(); else play();
    });
    listen(lessMotion, function () {
      if (lessMotion.matches) { pause(); draw(); } else stir(FIRST_BREATH);
    });

    measure();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
    stir(FIRST_BREATH);
  })();


  /* ---------- 3. films ---------- */

  // Takes a <figure class="film"> (see the film macro) and gives its Play button and progress bar their work.
  // Films never start by themselves. If a film cannot be played this way, the browser's own controls take over.
  // The atlas makes its films as they are needed and hands each one to this function.
  CASA.film = function (figure) {
    var video = figure.querySelector('video');
    var source = video && video.querySelector('source');
    var button = figure.querySelector('.film__play');
    var bar = figure.querySelector('.film__bar');
    if (!video || !button || figure.hasAttribute('data-ready')) return video;
    figure.setAttribute('data-ready', '');

    var tick = 0;
    function giveUp() { video.controls = true; button.hidden = true; if (bar) bar.hidden = true; }
    function progress() {
      cancelAnimationFrame(tick);
      if (bar && video.duration) bar.style.transform = 'scaleX(' + (video.currentTime / video.duration).toFixed(4) + ')';
      if (!video.paused) tick = requestAnimationFrame(progress);
    }
    function sync() {
      var playing = !video.paused && !video.ended;
      figure.classList.toggle('is-playing', playing);
      button.textContent = playing ? 'Pause' : 'Play';
      progress();
    }
    function toggle() {
      if (button.hidden) return;
      if (!video.paused) { video.pause(); return; }
      var started = video.play();
      if (started && started.catch) started.catch(function (error) { if (error && error.name === 'NotSupportedError') giveUp(); });
    }
    video.controls = false;
    button.hidden = false;
    button.addEventListener('click', toggle);
    video.addEventListener('click', toggle);
    video.addEventListener('play', sync);
    video.addEventListener('pause', sync);
    video.addEventListener('error', giveUp);
    if (source) source.addEventListener('error', giveUp);
    if (outOfSight) outOfSight.observe(figure);
    return video;
  };
  // A film stops when it is scrolled out of sight,
  var outOfSight = 'IntersectionObserver' in window && new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      var video = entry.target.querySelector('video');
      if (!entry.isIntersecting && video && !video.paused) video.pause();
    });
  });
  // and only one film on a page plays at a time (two that share a figure, as on the Research page, count as one).
  // The other is stopped with its own Pause button where it has one, so that its button and bar keep in step.
  document.addEventListener('play', function (event) {
    var figure = event.target.closest && event.target.closest('figure');
    each(document.querySelectorAll('video'), function (other) {
      if (other === event.target || other.paused || (figure && figure.contains(other))) return;
      var theirs = other.closest('figure');
      var button = theirs && theirs.querySelector('.film__play');
      if (button && !button.hidden && button.offsetParent) button.click(); else other.pause();
    });
  }, true);
  each(document.querySelectorAll('.film'), CASA.film);


  /* ---------- 4. figures that open larger ---------- */

  // A link marked data-zoom (see the plate macro) leads to the full picture. Here it opens over the page
  // instead, in a <dialog> that is made once, the first time it is wanted: the picture as wide as the sheet,
  // to be scrolled if it is taller. Escape, the cross, or a click outside the sheet closes it.
  // On a phone, upright or on its side, a sheet could show it no larger, so there the link opens the picture
  // itself, to be pinched and moved about. That is also what the link does without this file.
  var viewer = null;

  function makeViewer() {
    viewer = document.createElement('dialog');
    viewer.className = 'viewer';
    viewer.innerHTML = '<div class="viewer__pan" tabindex="0"><img class="viewer__img" alt=""></div><p class="viewer__words"></p>' +
      '<form method="dialog"><button class="viewer__close" type="submit" autofocus><span class="sr-only">Close</span></button></form>';
    viewer.addEventListener('click', function (event) { if (event.target === viewer) viewer.close(); });
    document.body.appendChild(viewer);
  }

  document.addEventListener('click', function (event) {
    var link = event.target.closest ? event.target.closest('a[data-zoom]') : null;
    if (!link || !window.HTMLDialogElement || window.innerWidth < 700 || window.innerHeight < 500) return;
    if (event.button || event.metaKey || event.ctrlKey || event.shiftKey) return;
    event.preventDefault();
    if (!viewer) makeViewer();
    var small = link.querySelector('img');
    var figure = link.closest('figure');
    var words = figure && figure.querySelector('figcaption p');
    var picture = viewer.querySelector('img');
    picture.src = link.href;
    picture.alt = small ? small.alt : '';
    if (small) {                        // its shape, so that the sheet is the right size before the picture arrives
      picture.setAttribute('width', small.getAttribute('width'));
      picture.setAttribute('height', small.getAttribute('height'));
    }
    viewer.setAttribute('aria-label', small ? small.alt : 'Figure');
    viewer.querySelector('.viewer__words').innerHTML = words ? words.innerHTML : '';
    viewer.showModal();
    viewer.querySelector('.viewer__pan').scrollTop = 0;
  });
  // The link works as a button, so the space bar opens it too.
  document.addEventListener('keydown', function (event) {
    var link = event.key === ' ' && event.target.closest ? event.target.closest('a[data-zoom]') : null;
    if (!link) return;
    event.preventDefault();
    link.click();
  });
})();
