import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

// TODO: put YOUR values here
const supabaseUrl = 'https://YOUR-PROJECT-ID.supabase.co';
const supabaseAnonKey = 'YOUR_ANON_PUBLIC_KEY';

const supabase = createClient(supabaseUrl, supabaseAnonKey);

const authSection = document.getElementById('auth');
const app = document.getElementById('app');
const loginForm = document.getElementById('login-form');
const authError = document.getElementById('auth-error');

function showApp() {
  authSection.style.display = 'none';
  app.style.display = 'block';
  // set footer year
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();
}

function showAuth() {
  authSection.style.display = 'flex'; // or 'block', depending on your CSS
  app.style.display = 'none';
}

async function checkSessionOnLoad() {
  const { data, error } = await supabase.auth.getUser();
  if (error) {
    console.error('Error getting user:', error);
    showAuth();
    return;
  }

  if (data.user) {
    showApp();
  } else {
    showAuth();
  }
}

loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  authError.textContent = '';

  const email = e.target.email.value;
  const password = e.target.password.value;

  // Log in with Supabase
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) {
    console.error(error);
    authError.textContent = error.message;
  } else {
    showApp();
  }
});

// Run on first load
checkSessionOnLoad();