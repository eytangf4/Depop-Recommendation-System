const authBtn = document.getElementById("auth-submit");
  let isSignup = false;

  document.getElementById("show-login").onclick = function () {
    isSignup = false;
    this.classList.add("active");
    document.getElementById("show-signup").classList.remove("active");
    authBtn.textContent = "Login";
  };

  document.getElementById("show-signup").onclick = function () {
    isSignup = true;
    this.classList.add("active");
    document.getElementById("show-login").classList.remove("active");
    authBtn.textContent = "Sign Up";
  };

  authBtn.onclick = function () {
    const username = document.getElementById("auth-username").value;
    const password = document.getElementById("auth-password").value;

    fetch(isSignup ? "/signup" : "/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    })
    .then(res => res.json())
    .then(data => {
      if (data.message) {
        document.getElementById("auth-status").textContent = data.message;
        document.getElementById("auth-status").classList.remove("d-none");
        setTimeout(() => window.location.href = data.redirect, 300);
      } else {
        alert(data.error);
      }
    });
  };