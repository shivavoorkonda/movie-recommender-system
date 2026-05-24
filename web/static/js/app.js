// app.js — CineAI Frontend SPA
'use strict';

// Set your production Render backend URL here!
// (e.g., 'https://movie-recommender-system.onrender.com')
const RENDER_BACKEND_URL = 'https://cineai-backend-f3sc.onrender.com'; 

const API = (window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1') || window.location.origin.startsWith('file:'))
  ? (window.location.origin.includes('5000') ? window.location.origin : 'http://localhost:5000')
  : (window.location.origin.includes('vercel.app') ? RENDER_BACKEND_URL : window.location.origin);

// ── Genre → emoji + colour ─────────────────────────────────────────────────
const GENRE_META = {
  'Action':       { emoji:'🎬', color:'#ff6b6b' },
  'Adventure':    { emoji:'🗺️', color:'#ffa94d' },
  'Animation':    { emoji:'🎨', color:'#a9e34b' },
  "Children's":   { emoji:'🧒', color:'#74c0fc' },
  'Comedy':       { emoji:'😂', color:'#ffd43b' },
  'Crime':        { emoji:'🔫', color:'#ff8787' },
  'Documentary':  { emoji:'🎥', color:'#a9e34b' },
  'Drama':        { emoji:'🎭', color:'#da77f2' },
  'Fantasy':      { emoji:'🔮', color:'#a9e34b' },
  'Film-Noir':    { emoji:'🕵️', color:'#ced4da' },
  'Horror':       { emoji:'👻', color:'#ff6b6b' },
  'Musical':      { emoji:'🎵', color:'#ffd43b' },
  'Mystery':      { emoji:'🕵️', color:'#74c0fc' },
  'Romance':      { emoji:'💘', color:'#ff6584' },
  'Sci-Fi':       { emoji:'🚀', color:'#00d4aa' },
  'Thriller':     { emoji:'🔪', color:'#ff8787' },
  'War':          { emoji:'⚔️',  color:'#868e96' },
  'Western':      { emoji:'🤠', color:'#ffa94d' },
  'unknown':      { emoji:'🎞️', color:'#6c63ff' },
};

function genreEmoji(genres) {
  if (!genres) return '🎞️';
  const g = genres.split('|')[0];
  return (GENRE_META[g] || GENRE_META['unknown']).emoji;
}
function genreColor(genres) {
  if (!genres) return '#6c63ff';
  const g = genres.split('|')[0];
  return (GENRE_META[g] || GENRE_META['unknown']).color;
}

// ── Fetch helper ────────────────────────────────────────────────────────────
async function apiFetch(url) {
  const r = await fetch(API + url);
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.error || e.message || r.statusText);
  }
  return r.json();
}

// ── Movie Card ──────────────────────────────────────────────────────────────
function makeCard(m, mode) {
  const col  = genreColor(m.genres);
  const emj  = genreEmoji(m.genres);
  let badge  = '';
  
  if (mode === 'hybrid' && m.hybrid_score != null)
    badge = `<span class="score-badge score-hybrid">Hybrid ${(m.hybrid_score*100).toFixed(0)}%</span>`;
  else if (mode === 'cf' && m.predicted_rating != null)
    badge = `<span class="score-badge score-cf">★ ${m.predicted_rating.toFixed(2)}</span>`;
  else if (mode === 'als' && m.predicted_rating != null)
    badge = `<span class="score-badge score-als">ALS ★ ${m.predicted_rating.toFixed(2)}</span>`;
  else if (mode === 'cb' && m.similarity_score != null)
    badge = `<span class="score-badge score-cb">Sim ${(m.similarity_score*100).toFixed(0)}%</span>`;
  else if (m.weighted_score != null)
    badge = `<span class="score-badge score-pop">★ ${(m.avg_rating||0).toFixed(1)}</span>`;
  else if (m.avg_rating != null && m.avg_rating > 0)
    badge = `<span class="score-badge score-cf">★ ${m.avg_rating.toFixed(1)}</span>`;

  const div = document.createElement('div');
  div.className = 'movie-card';
  const displayTitle = m.title ? (m.year ? `${m.title} (${m.year})` : m.title) : 'Unknown';
  div.innerHTML = `
    <div class="card-poster" style="background:linear-gradient(135deg,${col}22,${col}44)">
      <div class="card-genre-bg" style="background:${col}"></div>
      <span style="font-size:3rem;position:relative;z-index:1">${emj}</span>
    </div>
    <div class="card-body">
      <div class="card-title">${displayTitle}</div>
      <div class="card-genres">${(m.genres||'').replace(/\|/g,' · ')}</div>
      <div class="card-score">${badge}</div>
    </div>`;
  div.addEventListener('click', () => App.openMovieModal(m.movie_id));
  return div;
}

// ── Skeleton placeholder creators ───────────────────────────────────────────
function createSkeletons(n = 6) {
  let html = '';
  for (let i = 0; i < n; i++) {
    html += `
      <div class="movie-card skeleton">
        <div class="card-poster skeleton-wave"></div>
        <div class="card-body">
          <div class="skeleton-text skeleton-wave" style="width: 80%; height: 1.25rem; margin-bottom: 0.5rem;"></div>
          <div class="skeleton-text skeleton-wave" style="width: 50%; height: 0.85rem; margin-bottom: 0.5rem;"></div>
          <div class="skeleton-text skeleton-wave" style="width: 30%; height: 1.1rem;"></div>
        </div>
      </div>`;
  }
  return html;
}

function createMiniSkeletons(n = 5) {
  let html = '';
  for (let i = 0; i < n; i++) {
    html += `
      <div class="compare-skeleton-item">
        <div class="skeleton-wave" style="width:30px; height:30px; border-radius:4px; flex-shrink:0;"></div>
        <div style="flex:1;">
          <div class="skeleton-wave" style="width: 90%; height: 0.9rem; margin-bottom: 4px; border-radius:2px;"></div>
          <div class="skeleton-wave" style="width: 50%; height: 0.7rem; border-radius:2px;"></div>
        </div>
      </div>`;
  }
  return html;
}

function renderRow(containerId, movies, mode) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  if (!movies || !movies.length) {
    el.innerHTML = '<div class="loader-row">No results found.</div>';
    return;
  }
  movies.forEach(m => el.appendChild(makeCard(m, mode)));
}

function renderGrid(containerId, movies, mode) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  if (!movies || !movies.length) {
    el.innerHTML = '<div class="loader-row">No results.</div>';
    return;
  }
  movies.forEach(m => el.appendChild(makeCard(m, mode)));
}

// ── Rating stars ────────────────────────────────────────────────────────────
function stars(avg) {
  const full = Math.round(avg);
  return '★'.repeat(full) + '☆'.repeat(5-full);
}

// ── Toast ───────────────────────────────────────────────────────────────────
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), 3000);
}

// ── Main App ────────────────────────────────────────────────────────────────
const App = {
  currentUser: null,
  currentTab: 'hybrid',
  recMode: 'single', // 'single' or 'compare'
  dashMode: 'stats', // 'stats' or 'fairness'
  stats: null,
  fairnessData: null,

  // ── Init ────────────────────────────────────────────────────────────────
  async init() {
    this.bindNav();
    this.bindSearch();
    this.bindTabs();
    await this.loadStats();
    await this.loadPopular();
    this.showView('home');
  },

  // ── Nav ─────────────────────────────────────────────────────────────────
  showView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById('view-' + name).classList.add('active');
    document.querySelectorAll('.nav-link').forEach(l => {
      l.classList.toggle('active', l.dataset.view === name);
    });
    if (name === 'dashboard') {
      if (this.dashMode === 'stats') {
        this.loadDashboard();
      } else {
        this.loadFairness();
      }
    }
  },

  bindNav() {
    document.querySelectorAll('.nav-link').forEach(l => {
      l.addEventListener('click', e => { e.preventDefault(); this.showView(l.dataset.view); });
    });
  },

  // ── Global search ────────────────────────────────────────────────────────
  bindSearch() {
    const inp  = document.getElementById('globalSearch');
    const drop = document.getElementById('searchDropdown');
    let timer;
    inp.addEventListener('input', () => {
      clearTimeout(timer);
      const q = inp.value.trim();
      if (!q) { drop.classList.add('hidden'); return; }
      timer = setTimeout(() => this.doGlobalSearch(q, drop), 300);
    });
    document.addEventListener('click', e => {
      if (!e.target.closest('.search-box')) drop.classList.add('hidden');
    });
  },

  async doGlobalSearch(q, drop) {
    try {
      const data = await apiFetch(`/api/search?q=${encodeURIComponent(q)}&n=8`);
      drop.innerHTML = '';
      if (!data.results.length) {
        drop.innerHTML = '<div class="search-item"><div class="search-item-title" style="color:#9090a8">No results</div></div>';
      } else {
        data.results.forEach(m => {
          const item = document.createElement('div');
          item.className = 'search-item';
          const itemTitle = m.title ? (m.year ? `${m.title} (${m.year})` : m.title) : 'Unknown';
          item.innerHTML = `<div class="search-item-title">${genreEmoji(m.genres)} ${itemTitle}</div>
            <div class="search-item-meta">${(m.genres||'').replace(/\|/g,' · ')} ${m.avg_rating ? '· ★'+m.avg_rating : ''}</div>`;
          item.addEventListener('click', () => {
            drop.classList.add('hidden');
            this.openMovieModal(m.movie_id);
          });
          drop.appendChild(item);
        });
      }
      drop.classList.remove('hidden');
    } catch(e) { console.error(e); }
  },

  // ── Tabs ────────────────────────────────────────────────────────────────
  bindTabs() {
    document.querySelectorAll('.tab').forEach(t => {
      t.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
        t.classList.add('active');
        this.currentTab = t.dataset.tab;
        if (this.currentUser) this.loadRecommendations(this.currentUser, this.currentTab);
      });
    });
  },

  // ── Stats ────────────────────────────────────────────────────────────────
  async loadStats() {
    try {
      const s = await apiFetch('/api/stats');
      this.stats = s;
      document.getElementById('statRatings').textContent = (s.n_ratings/1000).toFixed(0) + 'K';
      document.getElementById('statUsers').textContent   = s.n_users;
      document.getElementById('statMovies').textContent  = s.n_movies;
      document.getElementById('statAvg').textContent     = s.avg_rating.toFixed(2) + ' ★';
    } catch(e) { console.error('Stats failed:', e); }
  },

  // ── Popular ──────────────────────────────────────────────────────────────
  async loadPopular() {
    document.getElementById('popularRow').innerHTML = createSkeletons(6);
    try {
      const d = await apiFetch('/api/popular?n=12');
      renderRow('popularRow', d.movies, 'popular');
    } catch(e) {
      document.getElementById('popularRow').innerHTML = '<div class="loader-row">Failed to load.</div>';
    }
  },

  // ── User Recommendations ─────────────────────────────────────────────────
  async loadUserView() {
    const uid = parseInt(document.getElementById('heroUserId').value);
    if (!uid || uid < 1 || uid > 943) { toast('Enter a user ID between 1 and 943'); return; }
    this.currentUser = uid;
    document.getElementById('userSection').style.display = 'block';
    
    // Fast scrolling
    document.getElementById('userSection').scrollIntoView({ behavior: 'smooth' });
    
    await this.loadRecommendations(uid, this.currentTab);
    await this.loadHistory(uid);
  },

  setRecMode(mode) {
    this.recMode = mode;
    document.getElementById('btnSingleView').classList.toggle('active', mode === 'single');
    document.getElementById('btnCompareView').classList.toggle('active', mode === 'compare');
    
    document.getElementById('recSingleContainer').classList.toggle('hidden', mode !== 'single');
    document.getElementById('recCompareContainer').classList.toggle('hidden', mode !== 'compare');
    
    if (this.currentUser) {
      this.loadRecommendations(this.currentUser, this.currentTab);
    }
  },

  async loadRecommendations(uid, mode) {
    if (this.recMode === 'compare') {
      // Comparison side-by-side view
      const cols = ['colHybridRow', 'colSGDRow', 'colALSRow', 'colCBRow'];
      cols.forEach(c => {
        document.getElementById(c).innerHTML = createMiniSkeletons(5);
      });
      
      try {
        const d = await apiFetch(`/api/compare/${uid}?n=5`);
        this.renderCompareList('colHybridRow', d.hybrid, 'hybrid');
        this.renderCompareList('colSGDRow', d.sgd, 'cf');
        this.renderCompareList('colALSRow', d.als || [], 'als');
        this.renderCompareList('colCBRow', d.cb, 'cb');
      } catch(e) {
        cols.forEach(c => {
          document.getElementById(c).innerHTML = `<div class="error-msg">${e.message}</div>`;
        });
      }
    } else {
      // Single model view
      const row = document.getElementById('recRow');
      row.innerHTML = createSkeletons(6);
      try {
        let url, recKey, recMode;
        if (mode === 'hybrid') {
          const alpha = document.getElementById('alphaSlider').value;
          url = `/api/recommend/${uid}?n=12&alpha=${alpha}`;
          recKey = 'recommendations'; recMode = 'hybrid';
        } else if (mode === 'cf') {
          url = `/api/cf/${uid}?n=12`;
          recKey = 'recommendations'; recMode = 'cf';
        } else if (mode === 'als') {
          url = `/api/als/${uid}?n=12`;
          recKey = 'recommendations'; recMode = 'als';
        } else {
          url = `/api/cb/${uid}?n=12`;
          recKey = 'recommendations'; recMode = 'cb';
        }
        const d = await apiFetch(url);
        renderRow('recRow', d[recKey], recMode);
      } catch(e) {
        row.innerHTML = `<div class="loader-row">${e.message}</div>`;
      }
    }
  },

  renderCompareList(containerId, items, mode) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';
    if (!items || !items.length) {
      el.innerHTML = '<div style="padding:1rem;color:var(--text2);font-size:0.8rem;">No recommendations</div>';
      return;
    }
    
    items.forEach((m, idx) => {
      const col = genreColor(m.genres);
      const emj = genreEmoji(m.genres);
      const itemEl = document.createElement('div');
      itemEl.className = 'compare-item';
      
      let badge = '';
      if (mode === 'hybrid' && m.score != null) {
        badge = `<span class="score-badge score-hybrid">${(m.score*100).toFixed(0)}%</span>`;
      } else if (mode === 'cf' && m.score != null) {
        badge = `<span class="score-badge score-cf">${m.score.toFixed(2)}</span>`;
      } else if (mode === 'als' && m.score != null) {
        badge = `<span class="score-badge score-als">${m.score.toFixed(2)}</span>`;
      } else if (mode === 'cb' && m.score != null) {
        badge = `<span class="score-badge score-cb">${(m.score*100).toFixed(0)}%</span>`;
      }
      
      itemEl.innerHTML = `
        <span class="compare-rank">${idx+1}</span>
        <div class="compare-avatar" style="background:linear-gradient(135deg, ${col}22, ${col}44); border: 1px solid ${col}66;">
          <span style="font-size:1.1rem;">${emj}</span>
        </div>
        <div class="compare-info">
          <div class="compare-title">${m.title}</div>
          <div class="compare-meta">${(m.genres||'').replace(/\|/g, ' · ')}</div>
        </div>
        <div class="compare-badge-wrap">${badge}</div>`;
        
      itemEl.addEventListener('click', () => this.openMovieModal(m.movie_id));
      el.appendChild(itemEl);
    });
  },

  async loadHistory(uid) {
    document.getElementById('historyRow').innerHTML = createSkeletons(6);
    try {
      const d = await apiFetch(`/api/user/${uid}/history`);
      const demo = d.demographics || {};
      
      const genderLabel = demo.gender === 'M' ? 'Male' : (demo.gender === 'F' ? 'Female' : '');
      document.getElementById('historyMeta').textContent =
        `User #${uid} · ${genderLabel} · Age ${demo.age||'?'} · ${demo.occupation||''} · ${d.n_ratings} ratings given`;
      
      // Map history back to a standard card structure
      const histData = d.history.slice(0, 12).map(m => ({
        movie_id: m.movie_id,
        title: m.title,
        genres: m.genres,
        year: m.year,
        avg_rating: m.user_rating
      }));
      
      renderRow('historyRow', histData, 'history');
    } catch(e) {
      document.getElementById('historyRow').innerHTML = `<div class="loader-row">${e.message}</div>`;
    }
  },

  // ── Explore ──────────────────────────────────────────────────────────────
  async exploreSearch() {
    const q = document.getElementById('exploreSearch').value.trim();
    if (!q) { toast('Enter a search term'); return; }
    document.getElementById('exploreResults').innerHTML = createSkeletons(12);
    document.getElementById('similarPanel').classList.add('hidden');
    try {
      const d = await apiFetch(`/api/search?q=${encodeURIComponent(q)}`);
      if (!d.total) {
        document.getElementById('exploreResults').innerHTML = '<div class="loader-row">No movies found.</div>';
        return;
      }
      renderGrid('exploreResults', d.results, 'popular');
    } catch(e) {
      document.getElementById('exploreResults').innerHTML = `<div class="loader-row">${e.message}</div>`;
    }
  },

  async showSimilar(movieId, title) {
    const panel = document.getElementById('similarPanel');
    document.getElementById('similarTitle').textContent = `Similar to: ${title}`;
    document.getElementById('similarRow').innerHTML = createSkeletons(6);
    panel.classList.remove('hidden');
    panel.scrollIntoView({ behavior: 'smooth' });
    try {
      const d = await apiFetch(`/api/similar/${movieId}?n=12`);
      renderRow('similarRow', d.similar, 'cb');
    } catch(e) {
      document.getElementById('similarRow').innerHTML = `<div class="loader-row">${e.message}</div>`;
    }
  },

  closeSimilar() {
    document.getElementById('similarPanel').classList.add('hidden');
  },

  // ── Dashboard ─────────────────────────────────────────────────────────────
  setDashMode(mode) {
    this.dashMode = mode;
    document.getElementById('btnDashStats').classList.toggle('active', mode === 'stats');
    document.getElementById('btnDashFairness').classList.toggle('active', mode === 'fairness');
    
    document.getElementById('dashStatsContainer').classList.toggle('hidden', mode !== 'stats');
    document.getElementById('dashFairnessContainer').classList.toggle('hidden', mode !== 'fairness');
    
    if (mode === 'stats') {
      this.loadDashboard();
    } else {
      this.loadFairness();
    }
  },

  async loadDashboard() {
    if (this.stats) { this.renderDashboard(this.stats); return; }
    try {
      const s = await apiFetch('/api/stats');
      this.stats = s;
      this.renderDashboard(s);
    } catch(e) { console.error(e); }
  },

  renderDashboard(s) {
    // Rating distribution bar chart
    const dist = s.rating_distribution;
    const maxV = Math.max(...Object.values(dist));
    const chartEl = document.getElementById('ratingDistChart');
    chartEl.innerHTML = '';
    ['1','2','3','4','5'].forEach(r => {
      const v = dist[r] || 0;
      const pct = maxV ? (v / maxV * 100) : 0;
      const colors = ['#ff6584','#ffa94d','#ffd43b','#a9e34b','#00d4aa'];
      const wrap = document.createElement('div');
      wrap.className = 'bar-wrap';
      wrap.innerHTML = `<div class="bar-val">${(v/1000).toFixed(1)}K</div>
        <div class="bar" style="height:${pct}%;background:${colors[+r-1]}"></div>
        <div class="bar-label">${r} ★</div>`;
      chartEl.appendChild(wrap);
    });

    // Genre chart (top 10)
    const gc = s.genre_counts;
    const sortedGenres = Object.entries(gc).sort((a,b)=>b[1]-a[1]).slice(0,10);
    const maxG = sortedGenres[0][1];
    const genreEl = document.getElementById('genreChart');
    genreEl.innerHTML = '';
    sortedGenres.forEach(([g, cnt]) => {
      const row = document.createElement('div');
      row.className = 'genre-row';
      row.innerHTML = `<div class="genre-name">${g}</div>
        <div class="genre-bar-wrap"><div class="genre-bar" style="width:${cnt/maxG*100}%"></div></div>
        <div class="genre-count">${cnt}</div>`;
      genreEl.appendChild(row);
    });

    // Top movies table
    const tbody = s.top_rated_movies.map((m,i) => {
      const titleYear = m.year ? ` (${m.year})` : '';
      return `
      <tr>
        <td>${i+1}</td>
        <td><strong>${m.title}${titleYear}</strong></td>
        <td style="color:#9090a8;font-size:0.8rem">${(m.genres||'').split('|').slice(0,2).join(' · ')}</td>
        <td><span class="stars">${stars(m.avg||0)}</span> ${(m.avg||0).toFixed(2)}</td>
        <td style="color:#9090a8">${(m.count||0).toLocaleString()}</td>
      </tr>`;
    }).join('');
    document.getElementById('topMoviesTable').innerHTML = `
      <table><thead><tr><th>#</th><th>Title</th><th>Genres</th><th>Avg Rating</th><th>Ratings</th></tr></thead>
      <tbody>${tbody}</tbody></table>`;
  },

  async loadFairness() {
    if (this.fairnessData) {
      this.renderFairness(this.fairnessData);
      return;
    }
    // Set loading
    document.getElementById('calibrationTableBody').innerHTML = `<tr><td colspan="4" style="text-align: center;"><div class="spinner" style="margin: 1rem auto;"></div></td></tr>`;
    
    try {
      const data = await apiFetch('/api/fairness');
      this.fairnessData = data;
      this.renderFairness(data);
    } catch(e) {
      document.getElementById('calibrationTableBody').innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--accent);">${e.message}</td></tr>`;
    }
  },

  renderFairness(d) {
    if (d.error) {
      document.getElementById('calibrationTableBody').innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--accent);">${d.error}</td></tr>`;
      return;
    }

    // Demographic Fairness
    const f = d.demographic_fairness || {};
    document.getElementById('fairMale').textContent   = f.male ? f.male.rmse.toFixed(4) : '-';
    document.getElementById('fairFemale').textContent = f.female ? f.female.rmse.toFixed(4) : '-';
    document.getElementById('fairAge25').textContent  = f.age_lt25 ? f.age_lt25.rmse.toFixed(4) : '-';
    document.getElementById('fairAge35').textContent  = f.age_25_35 ? f.age_25_35.rmse.toFixed(4) : '-';
    document.getElementById('fairAge50').textContent  = f.age_35_50 ? f.age_35_50.rmse.toFixed(4) : '-';
    document.getElementById('fairAgePl').textContent  = f.age_50plus ? f.age_50plus.rmse.toFixed(4) : '-';

    // Popularity Bias
    const p = d.popularity_bias || {};
    document.getElementById('giniVal').textContent     = p.gini_coefficient ? p.gini_coefficient.toFixed(4) : '-';
    document.getElementById('coverVal').textContent    = p.catalogue_coverage ? p.catalogue_coverage.toFixed(1) + '%' : '-%';
    document.getElementById('longTailVal').textContent = p.long_tail_ratio ? p.long_tail_ratio.toFixed(1) + '%' : '-%';

    // Calibration
    const c = d.calibration || {};
    const buckets = c.buckets || {};
    const rowsHtml = Object.keys(buckets).sort().map(rating => {
      const b = buckets[rating];
      const dev = b.calibration_error;
      const classDev = dev < 0.15 ? 'good' : (dev < 0.3 ? 'warn' : 'bad');
      return `
        <tr>
          <td><strong>${rating} ★</strong></td>
          <td>${b.mean_predicted.toFixed(3)}</td>
          <td><span class="deviation-indicator ${classDev}">${dev > 0 ? '+' : ''}${dev.toFixed(3)}</span></td>
          <td style="color: var(--text2);">${b.n.toLocaleString()}</td>
        </tr>`;
    }).join('');
    
    document.getElementById('calibrationTableBody').innerHTML = rowsHtml || `<tr><td colspan="4" style="text-align: center;">No calibration data found.</td></tr>`;
  },

  // ── Movie Modal ──────────────────────────────────────────────────────────
  async openMovieModal(movieId) {
    document.getElementById('movieModal').classList.remove('hidden');
    document.getElementById('modalContent').innerHTML = '<div class="spinner" style="margin:2rem auto"></div>';
    
    try {
      const promises = [
        apiFetch(`/api/movies/${movieId}`),
        apiFetch(`/api/similar/${movieId}?n=4`)
      ];

      // Add confidence endpoint request if user is active
      if (this.currentUser) {
        promises.push(apiFetch(`/api/confidence?user=${this.currentUser}&movie=${movieId}`));
      }

      const results = await Promise.all(promises);
      const m = results[0];
      const sim = results[1];
      const conf = this.currentUser ? results[2] : null;

      const dist = m.rating_distribution || {};
      const maxD = Math.max(...Object.values(dist).map(Number));
      const miniBar = ['1','2','3','4','5'].map(r => {
        const v = dist[r] || 0;
        const pct = maxD ? v / maxD * 100 : 0;
        const cols = ['#ff6584','#ffa94d','#ffd43b','#a9e34b','#00d4aa'];
        return `<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:3px">
          <div style="font-size:0.65rem;color:#9090a8">${v}</div>
          <div style="width:100%;height:${Math.max(pct,2)}%;background:${cols[+r-1]};border-radius:3px 3px 0 0;min-height:3px"></div>
          <div style="font-size:0.65rem;color:#9090a8">${r}★</div></div>`;
      }).join('');

      const simCards = (sim.similar || []).map(s => `
        <div style="font-size:0.78rem;padding:0.4rem 0.6rem;background:var(--bg);border-radius:6px;border:1px solid var(--border);cursor:pointer"
             onclick="App.closeModal();App.openMovieModal(${s.movie_id})">
          ${genreEmoji(s.genres)} ${s.title}
        </div>`).join('');

      // Confidence Interval Widget
      let confidenceHtml = '';
      if (conf) {
        const stdPct = Math.min(100, Math.max(0, (1 - conf.uncertainty / 1.5) * 100)); // Normalize uncertainty
        const certaintyLabel = conf.uncertainty < 0.25 ? 'High Confidence' : (conf.uncertainty < 0.5 ? 'Moderate Confidence' : 'Low Confidence (Uncertain)');
        const confidenceColor = conf.uncertainty < 0.25 ? '#3fb950' : (conf.uncertainty < 0.5 ? '#ffa657' : '#f78166');
        
        confidenceHtml = `
          <div class="confidence-widget" style="margin-top: 1rem; padding: 0.8rem; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.4rem;">
              <span style="font-size:0.8rem; color:var(--text2);">Model Prediction Quality:</span>
              <strong style="font-size:0.8rem; color:${confidenceColor};">${certaintyLabel}</strong>
            </div>
            <div style="display:flex; align-items:center; gap: 10px; margin-bottom: 0.5rem;">
              <div style="font-size:1.4rem; font-weight:700; color:var(--accent);">★ ${conf.mean_prediction.toFixed(2)}</div>
              <div style="flex:1;">
                <div style="font-size:0.7rem; color:var(--text2); display:flex; justify-content:space-between;">
                  <span>Confidence Interval (95%)</span>
                  <span>[${conf.ci_lower.toFixed(1)} - ${conf.ci_upper.toFixed(1)}]</span>
                </div>
                <div style="width:100%; height:6px; background:var(--border); border-radius:3px; overflow:hidden; margin-top:2px;">
                  <div style="width:${stdPct}%; height:100%; background:${confidenceColor}; border-radius:3px;"></div>
                </div>
              </div>
            </div>
            <div style="font-size:0.7rem; color:#9090a8; text-align:right;">Uncertainty (Std Dev): ±${conf.uncertainty.toFixed(3)}</div>
          </div>`;
      }

      document.getElementById('modalContent').innerHTML = `
        <div class="modal-title">${m.title}</div>
        <div class="modal-genres">${(m.genres||'').replace(/\|/g,' · ')}</div>
        <div class="modal-stats">
          <div class="modal-stat"><div class="val">${m.avg_rating||0}</div><div class="lbl">Avg Rating</div></div>
          <div class="modal-stat"><div class="val">${(m.n_ratings||0).toLocaleString()}</div><div class="lbl">Ratings</div></div>
          <div class="modal-stat"><div class="val">${m.year||'N/A'}</div><div class="lbl">Year</div></div>
        </div>
        ${confidenceHtml}
        <div style="font-size:0.8rem;color:#9090a8;margin: 1rem 0 0.5rem">Rating distribution</div>
        <div class="mini-bar-chart" style="height:70px">${miniBar}</div>
        ${sim.similar && sim.similar.length ? `
          <div style="font-size:0.8rem;color:#9090a8;margin:1rem 0 0.5rem">Similar movies</div>
          <div style="display:flex;flex-direction:column;gap:0.4rem">${simCards}</div>` : ''}
        <div style="margin-top:1.25rem;display:flex;gap:0.75rem;flex-wrap:wrap">
          <button class="btn btn-primary" onclick="App.closeModal();App.showView('explore');document.getElementById('exploreSearch').value='';App.showSimilar(${m.movie_id},'${m.title.replace(/'/g,"\\'")}')">
            Find Similar</button>
          ${this.currentUser ? `<button class="btn btn-ghost" onclick="App.closeModal()">Back</button>` : ''}
        </div>`;
    } catch(e) {
      document.getElementById('modalContent').innerHTML = `<p style="color:#ff6584">Error: ${e.message}</p>`;
    }
  },

  closeModal() {
    document.getElementById('movieModal').classList.add('hidden');
  },

  // ── Alpha slider re-apply ────────────────────────────────────────────────
  applyAlpha() {
    if (this.currentUser && this.currentTab === 'hybrid' && this.recMode === 'single') {
      this.loadRecommendations(this.currentUser, 'hybrid');
    }
  },
};

// Alpha slider live apply on mouseup
document.getElementById('alphaSlider').addEventListener('change', () => App.applyAlpha());

// Enter key in explore search
document.getElementById('exploreSearch').addEventListener('keydown', e => {
  if (e.key === 'Enter') App.exploreSearch();
});

// Close modal on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    App.closeModal();
    document.getElementById('searchDropdown').classList.add('hidden');
  }
});

// Boot
App.init();
