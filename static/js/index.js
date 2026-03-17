window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: false,
			infinite: false,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    bulmaSlider.attach();

    // Auto-pause/play so only one vertical results video plays at a time.
    (function setupSinglePlayingResultsVideo() {
      const videos = Array.from(document.querySelectorAll('.results-vertical video'));
      if (!videos.length) return;

      const pauseAllExcept = (active) => {
        for (const v of videos) {
          if (v !== active && !v.paused) v.pause();
        }
      };

      // If the user manually hits play, pause all others.
      for (const v of videos) {
        v.addEventListener('play', () => pauseAllExcept(v));
      }

      // Play the most-visible video while scrolling.
      const io = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          const v = entry.target;
          if (entry.isIntersecting) {
            pauseAllExcept(v);
            // Autoplay is allowed because videos are muted; ignore failures.
            v.play().catch(() => {});
          } else {
            v.pause();
          }
        }
      }, { threshold: 0.6 });

      for (const v of videos) io.observe(v);

      document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          for (const v of videos) v.pause();
        }
      });
    })();

    // Pause teaser video when scrolled out of view.
    (function setupTeaserPauseOnScroll() {
      const teaserVideo = document.querySelector('.hero.teaser video');
      if (!teaserVideo) return;

      const io = new IntersectionObserver((entries) => {
        const entry = entries[0];
        if (!entry) return;

        if (entry.isIntersecting) {
          // muted autoplay should be allowed; ignore failures.
          teaserVideo.play().catch(() => {});
        } else {
          teaserVideo.pause();
        }
      }, { threshold: 0.25 });

      io.observe(teaserVideo);
    })();

})
