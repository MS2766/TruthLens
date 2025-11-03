document.addEventListener('DOMContentLoaded', function(){
  // mark page as loaded
  document.body.classList.add('loaded');

  // animate progress bars
  document.querySelectorAll('.progress-bar').forEach(function(bar){
    try {
      const targetPercent = parseFloat(bar.getAttribute('data-progress'));
      if (isNaN(targetPercent)) return;

      // apply color based on value
      if (targetPercent >= 75) bar.classList.add('bg-success');
      else if (targetPercent >= 45) bar.classList.add('bg-warning');
      else bar.classList.add('bg-danger');

      bar.style.setProperty('--progress', targetPercent + '%');
      bar.classList.add('animate');
    } catch(e){ console.warn(e); }
  });

  // animate percent numbers
  document.querySelectorAll('[data-countup]').forEach(function(el){
    let end = parseFloat(el.getAttribute('data-countup'));
    let duration = 900;
    let start = 0;
    let frameRate = 30;
    let steps = Math.ceil(duration / frameRate);
    let stepVal = end / steps;
    let current = 0;

    let interval = setInterval(() => {
      current += stepVal;
      if (current >= end) {
        el.textContent = end.toFixed(1) + '%';
        clearInterval(interval);
      } else {
        el.textContent = current.toFixed(1) + '%';
      }
    }, frameRate);
  });

  // button click scale animation
  document.querySelectorAll('button').forEach(function(btn){
    btn.addEventListener('click', function(){
      btn.classList.add('pressed');
      setTimeout(()=>btn.classList.remove('pressed'), 200);
    });
  });
});
