// app.js — CineAI Frontend SPA
'use strict';

const API = '';   // same origin

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
    throw new Error(e.message || r.statusText);
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
  stats: null,

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
    if (name === 'dashboard' && !this.stats) this.loadDashboard();
    else if (name === 'dashboard' && this.stats) this.renderDashboard(this.stats);
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
      document.getElementById('statRatings').textContent = (s.total_ratings/1000).toFixed(0) + 'K';
      document.getElementById('statUsers').textContent   = s.total_users;
      document.getElementById('statMovies').textContent  = s.total_movies;
      document.getElementById('statAvg').textContent     = s.avg_rating.toFixed(2) + ' ★';
    } catch(e) { console.error('Stats failed:', e); }
  },

  // ── Popular ──────────────────────────────────────────────────────────────
  async loadPopular() {
    document.getElementById('popularRow').innerHTML = '<div class="loader-row"><div class="spinner"></div></div>';
    try {
      const d = await apiFetch('/api/popular?n=12');
      renderRow('popularRow', d.recommendations, 'popular');
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
    document.getElementById('userSection').scrollIntoView({ behavior: 'smooth' });
    this.currentTab = 'hybrid';
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab==='hybrid'));
    await this.loadRecommendations(uid, 'hybrid');
    await this.loadHistory(uid);
  },

  async loadRecommendations(uid, mode) {
    const row = document.getElementById('recRow');
    row.innerHTML = '<div class="loader-row"><div class="spinner"></div></div>';
    try {
      let url, recKey, recMode;
      if (mode === 'hybrid') {
        const alpha = document.getElementById('alphaSlider').value;
        url = `/api/recommend/${uid}?n=12&alpha=${alpha}`;
        recKey = 'recommendations'; recMode = 'hybrid';
      } else if (mode === 'cf') {
        url = `/api/cf/${uid}?n=12`;
        recKey = 'recommendations'; recMode = 'cf';
      } else {
        url = `/api/cb/${uid}?n=12`;
        recKey = 'recommendations'; recMode = 'cb';
      }
      const d = await apiFetch(url);
      renderRow('recRow', d[recKey], recMode);
    } catch(e) {
      row.innerHTML = `<div class="loader-row">${e.message}</div>`;
    }
  },

  async loadHistory(uid) {
    document.getElementById('historyRow').innerHTML = '<div class="loader-row"><div class="spinner"></div></div>';
    try {
      const d = await apiFetch(`/api/user/${uid}/history?n=12`);
      const info = d.user_info || {};
      document.getElementById('historyMeta').textContent =
        `User #${uid} · ${info.gender||''} · Age ${info.age||'?'} · ${info.occupation||''} · ${d.n_total} total ratings`;
      renderRow('historyRow', d.history.map(m => ({...m, avg_rating: m.rating})), 'cf');
    } catch(e) {
      document.getElementById('historyRow').innerHTML = `<div class="loader-row">${e.message}</div>`;
    }
  },

  // ── Explore ──────────────────────────────────────────────────────────────
  async exploreSearch() {
    const q = document.getElementById('exploreSearch').value.trim();
    if (!q) { toast('Enter a search term'); return; }
    document.getElementById('exploreResults').innerHTML = '<div class="loader-row"><div class="spinner"></div></div>';
    document.getElementById('similarPanel').classList.add('hidden');
    try {
      const d = await apiFetch(`/api/search?q=${encodeURIComponent(q)}&n=24`);
      if (!d.count) {
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
    document.getElementById('similarRow').innerHTML = '<div class="loader-row"><div class="spinner"></div></div>';
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

  // ── Movie Modal ──────────────────────────────────────────────────────────
  async openMovieModal(movieId) {
    document.getElementById('movieModal').classList.remove('hidden');
    document.getElementById('modalContent').innerHTML = '<div class="spinner" style="margin:2rem auto"></div>';
    try {
      const [m, sim] = await Promise.all([
        apiFetch(`/api/movies/${movieId}`),
        apiFetch(`/api/similar/${movieId}?n=4`),
      ]);

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

      document.getElementById('modalContent').innerHTML = `
        <div class="modal-title">${m.title}</div>
        <div class="modal-genres">${(m.genres||'').replace(/\|/g,' · ')}</div>
        <div class="modal-stats">
          <div class="modal-stat"><div class="val">${m.avg_rating||0}</div><div class="lbl">Avg Rating</div></div>
          <div class="modal-stat"><div class="val">${(m.n_ratings||0).toLocaleString()}</div><div class="lbl">Ratings</div></div>
          <div class="modal-stat"><div class="val">${m.year||'N/A'}</div><div class="lbl">Year</div></div>
        </div>
        <div style="font-size:0.8rem;color:#9090a8;margin-bottom:0.5rem">Rating distribution</div>
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
    if (this.currentUser && this.currentTab === 'hybrid') {
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
