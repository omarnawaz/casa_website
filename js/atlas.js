/* CASA: the atlas on the home page.
   Everything here is drawn from window.ATLAS: the world in dots, the inset of Britain and its
   neighbours, a line from Cardiff to every place, the list of names, and one story for each place.
   window.ATLAS is js/atlas-data.js, which build.py writes from data/places.yaml on every build.
   To add a place or change what one says, edit data/places.yaml, not this file.

   Pointing at a name (or moving to it with the keyboard) shows its story at once; clicking it keeps it.
   The arrow keys walk the list of names. Without this file, or if anything in it goes wrong, the page
   shows the places as a plain list. */
(function () {
  'use strict';

  var A = window.ATLAS;
  var globe = document.querySelector('.globe');
  if (!globe) return;
  if (!A) { globe.classList.add('is-plain'); return; }   // no data: the plain list stays, laid out as it is without JavaScript

  var list = globe.querySelector('.index');
  var plain = list.innerHTML;         // the plain list as the page arrived with it, in case drawing the map fails
  try {
    drawAtlas();
  } catch (error) {
    list.innerHTML = plain;
    globe.classList.add('is-plain');
    throw error;                      // so that it still shows in the browser's console
  }

  function drawAtlas() {
    var NS = 'http://www.w3.org/2000/svg';
    var WIDE = [20, 0, 932, 437.4];     // the world with the empty Pacific, where the inset sits
    var TIGHT = [143, 0, 809, 437.4];   // the world cropped closer, for small screens
    var BOW = 0.27;                     // how far a line from Cardiff bows towards the pole

    var calm = window.matchMedia('(prefers-reduced-motion: reduce)');
    var corner = window.matchMedia('(min-width: 900px)');    // the inset sits in a corner of the world
    var beside = window.matchMedia('(min-width: 1200px)');   // the names stand beside the map

    var worldBox = globe.querySelector('.world');
    var insetBox = globe.querySelector('.inset');
    var wrap = globe.querySelector('.index-wrap');
    var card = globe.querySelector('.card');
    var storyBox = globe.querySelector('.stories');

    var byId = {};
    A.places.forEach(function (p) { byId[p.id] = p; });
    var order = A.order.map(function (id) { return byId[id]; });
    var ids = A.order.slice();
    var home = byId[A.home] || byId.cardiff;   // the base: the lines start from it
    var chosen = null;      // the place that has been clicked
    var shown = null;       // the place whose story is showing: the one pointed at, or else the chosen one
    var peek = null;        // where a passing look came from: 'list', 'keys' or 'map'

    function each(nodes, fn) { Array.prototype.forEach.call(nodes, fn); }
    function draw(name, attrs, parent) {
      var node = document.createElementNS(NS, name);
      for (var key in attrs) node.setAttribute(key, attrs[key]);
      if (parent) parent.appendChild(node);
      return node;
    }
    function make(tag, cls, text) {
      var node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text != null) node.textContent = text;
      return node;
    }
    // A sentence from data/places.yaml may hold a little HTML, such as NO<sub>2</sub>.
    function sentence(tag, text) {
      var node = document.createElement(tag);
      node.innerHTML = curly(text);
      return node;
    }
    function curly(text) { return text.replace(/'/g, '’'); }
    function style(place) { return place.kind === 'talk' ? 'ring' : 'solid'; }   // two styles only
    function dotPath(dots) {
      var d = '';
      for (var i = 0; i < dots.length; i++) d += 'M' + dots[i][0] + ' ' + dots[i][1] + 'h0';
      return d;
    }
    function arcPath(x1, y1, x2, y2) {
      var nx = -(y2 - y1), ny = x2 - x1;            // a normal to the straight line between the two
      if (ny > 0) { nx = -nx; ny = -ny; }           // the one that points up the page
      return 'M' + x1 + ' ' + y1 + 'Q' + ((x1 + x2) / 2 + nx * BOW).toFixed(1) + ' ' +
        ((y1 + y2) / 2 + ny * BOW).toFixed(1) + ' ' + x2 + ' ' + y2;
    }


    /* ----- the world ----- */

    var world = draw('svg', { viewBox: TIGHT.join(' '), focusable: 'false' }, worldBox);
    var frame = A.inset.frameOnWorld;
    draw('rect', { 'class': 'frame', x: frame.x, y: frame.y, width: frame.w, height: frame.h, rx: 1.5 }, world);
    draw('path', { 'class': 'land', d: dotPath(A.dots) }, world);
    draw('rect', { 'class': 'frame-line', x: frame.x, y: frame.y, width: frame.w, height: frame.h, rx: 1.5 }, world);

    var marks = [];                       // every mark, on the world and in the inset
    var worldMarks = [], insetMarks = [];

    function addMark(place, layer, into, x, y, i) {
      var el = make('span', 'mark mark--' + style(place) + (place === home ? ' mark--hub' : ''));
      el.setAttribute('data-for', place.id);
      el.style.setProperty('--i', i);
      el.appendChild(make('i'));
      layer.appendChild(el);
      var mark = { id: place.id, el: el, x: x, y: y, dx: 0, dy: 0 };
      marks.push(mark);
      into.push(mark);
      return mark;
    }

    var far = order.filter(function (p) { return !p.inset; });
    var arcs = draw('g', {}, world);
    far.forEach(function (p, i) {
      var arc = draw('path', {
        'class': 'arc arc--' + style(p), 'data-for': p.id, pathLength: 1,
        d: arcPath(home.x, home.y, p.x, p.y)
      }, arcs);
      arc.style.setProperty('--i', i);
    });

    var worldLayer = make('div', 'marks');
    worldBox.appendChild(worldLayer);
    far.forEach(function (p, i) { addMark(p, worldLayer, worldMarks, p.x, p.y, i); });
    addMark(home, worldLayer, worldMarks, home.x, home.y, 0);   // one mark for Cardiff: the lines start from it


    /* ----- the inset: Britain and its neighbours, magnified ----- */

    var IV = A.inset.viewBox;
    var inset = draw('svg', { viewBox: IV.join(' '), focusable: 'false' }, insetBox);
    draw('path', { 'class': 'coast', d: A.inset.land }, inset);
    draw('path', { 'class': 'land', d: dotPath(A.inset.dots) }, inset);
    var near = A.inset.places.map(function (id) { return byId[id]; });
    var insetArcs = draw('g', {}, inset);
    near.forEach(function (p, i) {
      if (p === home) return;
      var arc = draw('path', {
        'class': 'arc arc--' + style(p), 'data-for': p.id, pathLength: 1,
        d: arcPath(home.ix, home.iy, p.ix, p.iy)
      }, insetArcs);
      arc.style.setProperty('--i', far.length + i);
    });
    var insetLayer = make('div', 'marks');
    insetBox.appendChild(insetLayer);
    near.forEach(function (p, i) { addMark(p, insetLayer, insetMarks, p.ix, p.iy, far.length + i); });


    /* ----- the names ----- */

    list.textContent = '';              // the plain list gives way to buttons
    var names = order.map(function (p) {
      var item = make('li');
      var button = make('button', p.name.length > 24 ? 'name name--long' : 'name');   // a long name may take two lines
      button.type = 'button';
      button.tabIndex = -1;             // Tab stops once in the list, at the chosen name (see choose); the arrow keys walk it
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute('data-id', p.id);
      var dot = make('i', 'dot dot--' + style(p));
      dot.setAttribute('aria-hidden', 'true');
      var text = make('span', 'name__text', p.name);
      text.setAttribute('data-name', p.name);
      button.appendChild(dot);
      button.appendChild(text);
      item.appendChild(button);
      list.appendChild(item);
      return { id: p.id, el: button };
    });

    var stepper = make('div', 'stepper');
    [[-1, 'Previous'], [1, 'Next']].forEach(function (pair) {
      var button = make('button', 'step', pair[1]);
      button.type = 'button';
      button.setAttribute('data-step', pair[0]);
      button.addEventListener('click', function () {
        var n = ids.length;
        choose(ids[(ids.indexOf(chosen) + pair[0] + n) % n]);
      });
      stepper.appendChild(button);
    });
    wrap.appendChild(stepper);


    /* ----- the stories ----- */

    var films = [];

    // A picture, with its caption under it if it has one. With enlarge: true in data/places.yaml it opens
    // larger when it is chosen, in the viewer that js/site.js makes for the figures on the Research page.
    function picture(pic) {
      var figure = make('figure', 'story__visual story__picture');
      var img = make('img');
      img.src = pic.src;
      img.alt = pic.alt;
      img.width = pic.width;
      img.height = pic.height;
      img.loading = 'lazy';
      if (pic.enlarge) {
        var link = make('a');
        link.href = pic.src;
        link.setAttribute('data-zoom', '');
        link.appendChild(img);
        link.appendChild(make('span', 'sr-only', 'Open this figure at full size.'));
        figure.appendChild(link);
      } else {
        figure.appendChild(img);
      }
      if (pic.caption) {
        var caption = make('figcaption');
        caption.appendChild(sentence('p', pic.caption));     // the viewer shows the words of figcaption p
        figure.appendChild(caption);
      }
      return figure;
    }

    function film(place) {
      var media = place.media, id = 'film-' + place.id;
      var figure = make('figure', 'film story__visual');
      figure.style.setProperty('--r', (media.width / media.height).toFixed(4));
      var box = make('div', 'film__frame');
      var video = document.createElement('video');
      video.width = media.width;
      video.height = media.height;
      video.preload = 'none';
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.setAttribute('playsinline', '');
      video.setAttribute('aria-labelledby', id);
      var source = document.createElement('source');
      source.src = media.video;
      source.type = 'video/mp4';
      video.appendChild(source);
      var bar = make('span', 'film__bar');
      var caption = sentence('figcaption', media.caption);
      caption.id = id;
      var button = make('button', 'film__play', 'Play');
      button.type = 'button';
      button.setAttribute('aria-describedby', id);
      box.appendChild(video);
      box.appendChild(bar);
      figure.appendChild(box);
      figure.appendChild(caption);
      figure.appendChild(button);

      if (window.CASA && window.CASA.film) {
        window.CASA.film(figure);          // js/site.js gives the button and the bar their work
      } else {                             // without it, the browser's own controls
        video.controls = true;
        button.hidden = true;
      }
      films.push({ id: place.id, video: video, poster: media.poster });
      return figure;
    }

    var stories = order.map(function (p) {
      var story = make('article', 'story');
      story.hidden = true;
      story.setAttribute('data-id', p.id);

      var head = make('header', 'story__head');
      head.appendChild(make('p', 'story__coords', p.coords));
      var name = make('h3', 'story__name');
      var parts = p.name.split(', ');
      name.appendChild(make('span', '', parts[0] + (parts[1] ? ',' : '')));
      if (parts[1]) {
        name.appendChild(document.createTextNode(' '));
        name.appendChild(make('span', '', parts[1]));
      }
      head.appendChild(name);
      story.appendChild(head);

      var words = make('div', 'story__words');
      words.appendChild(sentence('p', p.text));
      if (p.link) {
        var more = make('p', 'story__more');
        var link = make('a', '', p.linkText || 'Read the paper');
        link.href = p.link;
        more.appendChild(link);
        words.appendChild(more);
      }
      story.appendChild(words);

      // A film or a picture: the same place in the story, and the same layout (.story--visual in site.css).
      // So that the two sides balance, the sentence is set larger when it is short, and the picture or film
      // larger when it has no caption under it. text_size and image_size in data/places.yaml overrule both.
      if (p.media || p.picture) {
        var length = words.textContent.length;
        var hasCaption = !!(p.media ? p.media.caption : p.picture.caption);
        story.classList.add('story--visual', 'story--text-' + (p.textSize || (length <= 140 ? 'large' : length <= 220 ? 'medium' : 'small')));
        story.classList.add('story--image-' + (p.imageSize || (hasCaption ? 'medium' : 'large')));
      }
      if (p.media) {
        story.classList.add('story--film');
        story.appendChild(film(p));
      } else if (p.picture) {
        story.classList.add('story--picture');
        story.appendChild(picture(p.picture));
      }
      storyBox.appendChild(story);
      names[ids.indexOf(p.id)].words = words.firstChild.textContent;   // the sentence without any HTML, to be read aloud
      return { id: p.id, el: story };
    });

    var spoken = make('p', 'sr-only');        // tells screen readers which story has been chosen
    spoken.setAttribute('aria-live', 'polite');
    card.appendChild(spoken);


    /* ----- showing and choosing ----- */

    // A film's still picture is fetched when its story is first shown, or when the map comes near the
    // window, and not with the page: most visitors see one story's film, if any.
    function fetchPosters(id) {
      films.forEach(function (f) { if ((!id || f.id === id) && !f.video.poster) f.video.poster = f.poster; });
    }

    function show(id) {
      if (id === shown || !byId[id]) return;
      shown = id;
      fetchPosters(id);
      stories.forEach(function (story) { story.el.hidden = story.id !== id; });
      films.forEach(function (f) { if (f.id !== id && !f.video.paused) f.video.pause(); });
      names.forEach(function (name) { name.el.classList.toggle('is-shown', name.id === id); });
      marks.forEach(function (mark) { mark.el.classList.toggle('is-on', mark.id === id); });
      each(globe.querySelectorAll('.arc'), function (arc) {
        arc.classList.toggle('is-on', arc.getAttribute('data-for') === id);
      });
      globe.classList.toggle('is-near', !!byId[id].inset);   // the two frames light up together
    }

    // Where the names are a column too long for its room, the list is scrolled until the name shows whole.
    function scrollTo(button) {
      var room = list.getBoundingClientRect(), box = button.getBoundingClientRect();
      if (box.top < room.top) list.scrollTop -= room.top - box.top + 3;
      else if (box.bottom > room.bottom) list.scrollTop += box.bottom - room.bottom + 3;
    }

    function keepInView(button, instantly) {
      if (list.scrollHeight > list.clientHeight + 1) {
        scrollTo(button);
        setTimeout(function () { scrollTo(button); }, 80);    // and again, in case turning bold gave it another line
      }
      if (list.scrollWidth <= list.clientWidth + 1) return;   // the rest only where the names are a row to swipe
      var left = button.getBoundingClientRect().left - list.getBoundingClientRect().left + list.scrollLeft -
        (list.clientWidth - button.offsetWidth) / 2;
      if (list.scrollTo) list.scrollTo({ left: left, behavior: instantly || calm.matches ? 'auto' : 'smooth' });
      else list.scrollLeft = left;
    }

    function choose(id, quietly) {
      if (!byId[id]) return;
      var changed = id !== chosen;
      chosen = id;
      peek = null;
      names.forEach(function (name) {
        name.el.setAttribute('aria-pressed', String(name.id === id));
        name.el.tabIndex = name.id === id ? 0 : -1;
      });
      marks.forEach(function (mark) { mark.el.classList.toggle('is-chosen', mark.id === id); });
      show(id);
      keepInView(names[ids.indexOf(id)].el, quietly);
      if (quietly || !changed) return;
      each(globe.querySelectorAll('.arc.is-run'), function (arc) { arc.classList.remove('is-run'); });
      var arc = globe.querySelector('.arc[data-for="' + id + '"]');
      if (arc && !calm.matches && !globe.classList.contains('will-arrive')) {   // the line runs out from Cardiff
        arc.getBoundingClientRect();
        arc.classList.add('is-run');
      }
      spoken.textContent = byId[id].name + '. ' + names[ids.indexOf(id)].words;
    }

    function look(id, from) { peek = from; show(id); }
    function lookAway(from) {
      if (peek !== from) return;
      peek = null;
      show(chosen);
    }

    function nameAt(target) { return target && target.closest ? target.closest('.name') : null; }

    list.addEventListener('mouseover', function (event) {
      var button = nameAt(event.target);
      if (button) look(button.getAttribute('data-id'), 'list');
    });
    list.addEventListener('mouseleave', function () { lookAway('list'); });
    list.addEventListener('focusin', function (event) {
      var button = nameAt(event.target), byKeyboard = true;
      if (!button) return;
      try { byKeyboard = button.matches(':focus-visible'); } catch (err) { /* older browsers: assume keyboard */ }
      if (byKeyboard) look(button.getAttribute('data-id'), 'keys');
    });
    list.addEventListener('focusout', function (event) {
      if (!list.contains(event.relatedTarget)) lookAway('keys');
    });
    list.addEventListener('click', function (event) {
      var button = nameAt(event.target);
      if (button) choose(button.getAttribute('data-id'));
    });
    list.addEventListener('keydown', function (event) {       // arrow keys walk the list
      var button = nameAt(event.target);
      if (!button || event.altKey || event.ctrlKey || event.metaKey) return;
      var i = ids.indexOf(button.getAttribute('data-id')), n = ids.length, to = -1;
      if (event.key === 'ArrowDown' || event.key === 'ArrowRight') to = (i + 1) % n;
      else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') to = (i - 1 + n) % n;
      else if (event.key === 'Home') to = 0;
      else if (event.key === 'End') to = n - 1;
      if (to < 0) return;
      event.preventDefault();
      names[to].el.focus();
      look(ids[to], 'keys');
    });


    /* ----- the marks on the maps: the nearest one to the pointer is the one in hand ----- */

    function nearest(box, among, x, y, reach) {
      var r = box.getBoundingClientRect(), best = null, bestD = reach * reach;
      among.forEach(function (mark) {
        var dx = r.left + mark.px - x, dy = r.top + mark.py - y, d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = mark; }
      });
      return best ? best.id : null;
    }

    function live(box, among) {
      box.addEventListener('pointermove', function (event) {
        if (event.pointerType === 'touch') return;
        var id = nearest(box, among, event.clientX, event.clientY, 24);
        box.style.cursor = id ? 'pointer' : '';
        if (id) look(id, 'map'); else lookAway('map');
      });
      box.addEventListener('pointerleave', function () {
        box.style.cursor = '';
        lookAway('map');
      });
      box.addEventListener('click', function (event) {
        var id = nearest(box, among, event.clientX, event.clientY, 32);
        if (!id) return;
        choose(id);
        bringStoryUp();
      });
    }
    live(worldBox, worldMarks);
    live(insetBox, insetMarks);

    // Below 1200px wide the story stands under the maps and the names. When a place is chosen on a map
    // and its story starts low on the screen, the page moves up so that the Previous and Next buttons and
    // the story are in view (the next place is a tap away there). Where the story is already in view,
    // and beside the maps on a wide screen, nothing moves.
    var stacked = window.matchMedia('(max-width: 1199.98px)');
    function bringStoryUp() {
      if (!stacked.matches) return;
      if (card.getBoundingClientRect().top < window.innerHeight * 0.55) return;
      var top = stepper.getBoundingClientRect().top + window.pageYOffset - 12;
      window.scrollTo({ top: top, behavior: calm.matches ? 'auto' : 'smooth' });
    }


    /* ----- sizes ----- */

    var view = null;

    function settle(among, w, h, x0, y0, vw, vh, apart) {
      among.forEach(function (mark) {
        mark.px = (mark.x - x0) / vw * w;
        mark.py = (mark.y - y0) / vh * h;
      });
      // Two marks never sit on top of one another (Cardiff and Port Talbot would, and on a phone
      // so would Houston and New Orleans): they step apart along the line between them, just far
      // enough to be told apart and chosen. A few passes, so that one step does not undo another.
      for (var pass = 0; apart && pass < 6; pass++) {
        for (var i = 0; i < among.length; i++) {
          for (var j = i + 1; j < among.length; j++) {
            var a = among[i], b = among[j];
            var dx = b.px - a.px, dy = b.py - a.py, gap = Math.sqrt(dx * dx + dy * dy) || 1;
            if (gap >= apart) continue;
            var push = (apart - gap) / 2;
            a.px -= dx / gap * push; a.py -= dy / gap * push;
            b.px += dx / gap * push; b.py += dy / gap * push;
          }
        }
      }
      among.forEach(function (mark) {
        mark.el.style.left = mark.px.toFixed(1) + 'px';
        mark.el.style.top = mark.py.toFixed(1) + 'px';
      });
    }

    function measure() {
      var next = corner.matches ? WIDE : TIGHT;
      if (next !== view) {
        view = next;
        world.setAttribute('viewBox', view.join(' '));
      }
      var size = parseFloat(getComputedStyle(insetMarks[0].el).getPropertyValue('--m')) || 12;
      settle(worldMarks, worldBox.clientWidth, worldBox.clientHeight, view[0], view[1], view[2], view[3], size + 3);
      settle(insetMarks, insetBox.clientWidth, insetBox.clientHeight, IV[0], IV[1], IV[2], IV[3], size + 5);
    }

    if (window.ResizeObserver) {
      var watch = new ResizeObserver(measure);
      watch.observe(worldBox);
      watch.observe(insetBox);
      watch.observe(storyBox);
    }
    // Where the names are a row to swipe, the chosen one is brought back to the middle when the
    // row changes width (a turned phone, or the lettering arriving), but never while it is being swiped.
    var lastWidth = window.innerWidth;
    function recentre() {
      if (chosen) keepInView(names[ids.indexOf(chosen)].el, true);
    }
    window.addEventListener('resize', function () {
      measure();
      if (window.innerWidth === lastWidth) return;
      lastWidth = window.innerWidth;
      recentre();
    });


    /* ----- begin ----- */

    choose(byId[A.start] ? A.start : ids[0], true);
    measure();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { measure(); recentre(); });

    if (window.IntersectionObserver) {
      var nearby = new IntersectionObserver(function (entries) {
        if (!entries[0].isIntersecting) return;
        nearby.disconnect();
        fetchPosters();
      }, { rootMargin: '200px' });
      nearby.observe(card);
    } else {
      fetchPosters();
    }

    // The lines draw out from Cardiff the first time the map comes into view.
    if (window.IntersectionObserver && !calm.matches) {
      globe.classList.add('will-arrive');
      var seen = new IntersectionObserver(function (entries) {
        if (!entries[0].isIntersecting) return;
        seen.disconnect();
        globe.classList.add('is-seen');
        setTimeout(function () { globe.classList.remove('will-arrive', 'is-seen'); }, 2400);
      }, { threshold: 0.3 });
      seen.observe(worldBox);
    }
  }
})();
