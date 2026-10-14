/* CASA: the cards on the Group page. Each portrait turns over to show a few sentences.
   Turning needs a pointer that can hover and room for the cards to sit side by side.
   Everywhere else (touch screens, narrow windows, no JavaScript) the sentences sit under each portrait. */
(function () {
  'use strict';

  var root = document.documentElement;
  var canTurn = window.matchMedia('(hover: hover) and (pointer: fine) and (min-width: 620px)');
  var people = Array.prototype.slice.call(document.querySelectorAll('.person'));
  var lastPointer = 'mouse';
  if (!people.length) return;

  function each(nodes, fn) { Array.prototype.forEach.call(nodes, fn); }

  // aria-pressed says which side is showing: true while the sentences are, however they came to be.
  function say(person) {
    var turned = person.classList.contains('is-open') ||
      (person.classList.contains('is-keyed') && !person.classList.contains('is-shut'));
    person.querySelector('.person__turn').setAttribute('aria-pressed', String(turned));
  }

  function close(person) {
    person.classList.remove('is-open');
    say(person);
  }

  function syncTurning() {
    var on = canTurn.matches;
    root.classList.toggle('flip', on);
    people.forEach(function (person) {
      person.querySelector('.person__turn').hidden = !on;
      if (!on) {
        close(person);
        person.classList.remove('is-keyed', 'is-shut');
      }
    });
  }

  people.forEach(function (person) {
    var button = person.querySelector('.person__turn');
    var links = person.querySelector('.person__links');
    var back = person.querySelector('.person__back');

    // The same links again on the other side of the card, in the same place, so a pointer can
    // reach them whichever side is showing. Keyboards and screen readers use the first set only.
    if (links && back) {
      var copy = links.cloneNode(true);
      copy.classList.add('person__links--back');
      copy.removeAttribute('aria-label');
      copy.setAttribute('aria-hidden', 'true');
      each(copy.querySelectorAll('a'), function (link) { link.tabIndex = -1; });
      back.appendChild(copy);

      // A link reached by keyboard is on the front, so the front has to be showing.
      links.addEventListener('focusin', function (e) {
        var byKeyboard = true;
        try { byKeyboard = e.target.matches(':focus-visible'); } catch (err) { /* older browsers: assume keyboard */ }
        if (!byKeyboard) return;
        close(person);
        person.classList.add('is-shut');
        say(person);
      });
      links.addEventListener('focusout', function () { person.classList.remove('is-shut'); say(person); });
    }

    person.addEventListener('pointerdown', function (e) { lastPointer = e.pointerType || 'mouse'; });
    person.addEventListener('pointerleave', function () {
      if (!links || !links.contains(document.activeElement)) person.classList.remove('is-shut');
    });

    // A click or key press turns the card over from whichever side is showing.
    // Turned by a click, it stays turned after the pointer or the focus moves on.
    button.addEventListener('click', function () {
      var shut = person.classList.contains('is-shut');
      var hovered = lastPointer === 'mouse' && person.matches(':hover');
      var showing = person.classList.contains('is-open') ||
        (!shut && (hovered || person.classList.contains('is-keyed')));
      if (showing) {
        close(person);
        person.classList.add('is-shut');
      } else {
        person.classList.remove('is-shut');
        person.classList.add('is-open');
      }
      say(person);
    });

    button.addEventListener('focus', function () {
      var byKeyboard = true;
      try { byKeyboard = button.matches(':focus-visible'); } catch (err) { /* older browsers: assume keyboard */ }
      if (byKeyboard) person.classList.add('is-keyed');
      say(person);
    });
    button.addEventListener('blur', function () {
      person.classList.remove('is-keyed', 'is-shut');
      say(person);
    });
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') people.forEach(close);
  });
  if (canTurn.addEventListener) canTurn.addEventListener('change', syncTurning);
  else if (canTurn.addListener) canTurn.addListener(syncTurning);
  syncTurning();
})();
