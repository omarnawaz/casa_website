/* CASA: the dataset page. One thing: the button that copies the citation.
   The button is hidden in the page and shown here, so without JavaScript it is simply not there. */
(function () {
  'use strict';

  var button = document.querySelector('[data-copy]');                     // <button data-copy="#citation">
  var paper = button && document.querySelector(button.getAttribute('data-copy'));
  var spoken = document.querySelector('[data-copied]');                   // read out by screen readers
  if (!button || !paper) return;
  var timer = 0;

  // The citation as plain text, on one line.
  function words() { return paper.textContent.replace(/\s+/g, ' ').trim(); }

  function say(sentence) { if (spoken) spoken.textContent = sentence; }

  // For a browser without navigator.clipboard, or one that refuses it.
  function byHand(text) {
    var box = document.createElement('textarea');
    box.value = text;
    box.setAttribute('readonly', '');
    box.style.cssText = 'position: fixed; top: -200px; left: 0; opacity: 0;';
    document.body.appendChild(box);
    box.select();
    var worked = false;
    try { worked = document.execCommand('copy'); } catch (error) { worked = false; }
    document.body.removeChild(box);
    button.focus();
    return worked;
  }

  // The button shows a tick for a moment, and a screen reader says what happened.
  function copied() {
    button.setAttribute('data-done', '');
    say('The citation has been copied.');
    window.clearTimeout(timer);
    timer = window.setTimeout(function () {
      button.removeAttribute('data-done');
      say('');
    }, 2600);
  }

  // If nothing works, the citation is selected, ready to be copied with the keyboard.
  function selected() {
    var range = document.createRange();
    range.selectNodeContents(paper);
    var selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    say('The citation could not be copied, so it has been selected instead.');
  }

  button.addEventListener('click', function () {
    var text = words();
    function fallBack() { if (byHand(text)) copied(); else selected(); }
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(copied, fallBack);
    else fallBack();
  });
  button.hidden = false;
})();
