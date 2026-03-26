/**
 * MMaDA-VLA Project Page JavaScript
 * Minimalist Academic Design
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initNavigation();
    initScrollAnimations();
    initSmoothScroll();
    initBarAnimations();
});

/**
 * Navigation functionality
 */
function initNavigation() {
    const nav = document.querySelector('.nav');
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    // Navbar scroll effect
    let lastScroll = 0;
    
    window.addEventListener('scroll', function() {
        const currentScroll = window.pageYOffset;
        
        // Add/remove scrolled class for styling
        if (currentScroll > 50) {
            nav.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.08)';
        } else {
            nav.style.boxShadow = 'none';
        }
        
        // Hide/show navbar on scroll
        if (currentScroll > lastScroll && currentScroll > 100) {
            nav.style.transform = 'translateY(-100%)';
        } else {
            nav.style.transform = 'translateY(0)';
        }
        
        lastScroll = currentScroll;
    });
    
    // Mobile menu toggle
    if (navToggle) {
        navToggle.addEventListener('click', function() {
            this.classList.toggle('active');
            
            // Create mobile menu if it doesn't exist
            let mobileMenu = document.querySelector('.nav-mobile');
            
            if (!mobileMenu) {
                mobileMenu = document.createElement('div');
                mobileMenu.className = 'nav-mobile';
                mobileMenu.innerHTML = navLinks.innerHTML;
                
                // Style the mobile menu
                mobileMenu.style.cssText = `
                    position: fixed;
                    top: 72px;
                    left: 0;
                    right: 0;
                    background: rgba(255, 255, 255, 0.98);
                    backdrop-filter: blur(20px);
                    padding: 24px;
                    border-bottom: 1px solid var(--border);
                    transform: translateY(-100%);
                    opacity: 0;
                    transition: all 0.3s ease;
                    z-index: 999;
                `;
                
                // Style mobile links
                const links = mobileMenu.querySelectorAll('a');
                links.forEach(link => {
                    link.style.cssText = `
                        display: block;
                        padding: 16px;
                        font-size: 1rem;
                        font-weight: 500;
                        color: var(--text-primary);
                        border-radius: 8px;
                        transition: all 0.2s ease;
                    `;
                    
                    link.addEventListener('mouseenter', () => {
                        link.style.background = 'var(--bg-secondary)';
                    });
                    
                    link.addEventListener('mouseleave', () => {
                        link.style.background = 'transparent';
                    });
                    
                    link.addEventListener('click', () => {
                        closeMobileMenu();
                    });
                });
                
                document.body.appendChild(mobileMenu);
            }
            
            // Toggle menu visibility
            if (this.classList.contains('active')) {
                mobileMenu.style.transform = 'translateY(0)';
                mobileMenu.style.opacity = '1';
                document.body.style.overflow = 'hidden';
            } else {
                closeMobileMenu();
            }
        });
    }
    
    function closeMobileMenu() {
        const mobileMenu = document.querySelector('.nav-mobile');
        const toggle = document.querySelector('.nav-toggle');
        
        if (mobileMenu) {
            mobileMenu.style.transform = 'translateY(-100%)';
            mobileMenu.style.opacity = '0';
        }
        
        if (toggle) {
            toggle.classList.remove('active');
        }
        
        document.body.style.overflow = '';
    }
    
    // Close mobile menu on window resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768) {
            closeMobileMenu();
        }
    });
}

/**
 * Scroll animations for sections
 */
function initScrollAnimations() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements for fade-in animation
    const animatedElements = document.querySelectorAll(
        '.section-header, .abstract-card, .feature, .step, .arch-card, ' +
        '.video-card, .result-block, .task-result-card, .ablation-table-wrapper, ' +
        '.citation-box, .paper-links'
    );
    
    animatedElements.forEach((el, index) => {
        el.classList.add('fade-in');
        el.style.transitionDelay = `${index * 0.05}s`;
        observer.observe(el);
    });
}

/**
 * Smooth scroll for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            if (href === '#') return;
            
            e.preventDefault();
            
            const target = document.querySelector(href);
            if (target) {
                const navHeight = document.querySelector('.nav').offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Animate progress bars when in view
 */
function initBarAnimations() {
    const bars = document.querySelectorAll('.bar-fill');
    
    const observerOptions = {
        root: null,
        threshold: 0.5
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                const width = bar.style.width;
                bar.style.width = '0';
                
                setTimeout(() => {
                    bar.style.width = width;
                }, 100);
                
                observer.unobserve(bar);
            }
        });
    }, observerOptions);
    
    bars.forEach(bar => observer.observe(bar));
}

/**
 * Copy citation to clipboard
 */
function copyCitation() {
    const citation = document.querySelector('.citation-box code').textContent;
    
    navigator.clipboard.writeText(citation).then(() => {
        const btn = document.querySelector('.copy-btn');
        const originalHTML = btn.innerHTML;
        
        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        btn.style.background = 'rgba(34, 197, 94, 0.3)';
        
        setTimeout(() => {
            btn.innerHTML = originalHTML;
            btn.style.background = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy citation:', err);
        
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = citation;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        const btn = document.querySelector('.copy-btn');
        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        setTimeout(() => {
            btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
        }, 2000);
    });
}

/**
 * Video placeholder click handler
 */
document.querySelectorAll('.video-placeholder').forEach(placeholder => {
    placeholder.addEventListener('click', function() {
        // Add play animation or modal functionality here
        this.style.transform = 'scale(0.98)';
        setTimeout(() => {
            this.style.transform = '';
        }, 150);
    });
});

/**
 * Parallax effect for hero background
 */
window.addEventListener('scroll', function() {
    const scrolled = window.pageYOffset;
    const heroBg = document.querySelector('.hero-bg');
    
    if (heroBg && scrolled < window.innerHeight) {
        heroBg.style.transform = `translateY(${scrolled * 0.3}px)`;
    }
});

/**
 * Active section highlighting in navigation
 */
function updateActiveNav() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        const navHeight = document.querySelector('.nav').offsetHeight;
        
        if (window.pageYOffset >= sectionTop - navHeight - 100) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.style.color = '';
        link.style.fontWeight = '';
        
        if (link.getAttribute('href') === `#${current}`) {
            link.style.color = 'var(--primary)';
            link.style.fontWeight = '600';
        }
    });
}

window.addEventListener('scroll', updateActiveNav);
window.addEventListener('load', updateActiveNav);


/**
 * Handle video loading states
 */
document.querySelectorAll('video').forEach(video => {
    video.addEventListener('loadeddata', function() {
        this.classList.add('loaded');
    });
    
    video.addEventListener('error', function() {
        console.error('Error loading video:', this.src);
    });
});

/**
 * Keyboard navigation support
 */
document.addEventListener('keydown', function(e) {
    // ESC key closes mobile menu
    if (e.key === 'Escape') {
        const mobileMenu = document.querySelector('.nav-mobile');
        if (mobileMenu && mobileMenu.style.opacity === '1') {
            document.querySelector('.nav-toggle').click();
        }
    }
});

/**
 * Preload critical resources
 */
window.addEventListener('load', function() {
    // Add loaded class to body for CSS transitions
    document.body.classList.add('loaded');
    
    // Animate hero elements
    const heroElements = document.querySelectorAll('.hero-title, .hero-subtitle, .authors, .hero-buttons, .hero-stats');
    heroElements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'all 0.6s ease';
        
        setTimeout(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, 100 + index * 100);
    });
});

// Expose copyCitation to global scope for HTML onclick attribute
window.copyCitation = copyCitation;

// Serial Work dropdown - prevent button click
window.addEventListener('load', function() {
    var btn = document.getElementById('serial-work-btn');
    if (btn) {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
        });
    }
});
