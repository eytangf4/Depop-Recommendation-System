document.addEventListener('DOMContentLoaded', function() {
    const authBtn = document.getElementById("auth-submit");
    const resetBtn = document.getElementById("reset-submit");
    const authForm = document.getElementById("auth-form");
    const resetForm = document.getElementById("reset-form");
    let isSignup = false;
  
    document.getElementById("show-login").onclick = function (e) {
      e.preventDefault();
      isSignup = false;
      this.classList.add("active");
      document.getElementById("show-signup").classList.remove("active");
      document.getElementById("show-reset").classList.remove("active");
      authForm.classList.remove("d-none");
      resetForm.classList.add("d-none");
      authBtn.textContent = "Login";
    };
  
    document.getElementById("show-signup").onclick = function (e) {
      e.preventDefault();
      isSignup = true;
      this.classList.add("active");
      document.getElementById("show-login").classList.remove("active");
      document.getElementById("show-reset").classList.remove("active");
      authForm.classList.remove("d-none");
      resetForm.classList.add("d-none");
      authBtn.textContent = "Sign Up";
    };

    document.getElementById("show-reset").onclick = function (e) {
      e.preventDefault();
      this.classList.add("active");
      document.getElementById("show-login").classList.remove("active");
      document.getElementById("show-signup").classList.remove("active");
      authForm.classList.add("d-none");
      resetForm.classList.remove("d-none");
    };
  
    authBtn.onclick = function (e) {
      e.preventDefault();
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

    resetBtn.onclick = function (e) {
      e.preventDefault();
      const username = document.getElementById("reset-username").value;
      const newPassword = document.getElementById("reset-new-password").value;
  
      if (!username || !newPassword) {
        alert("Please enter both username and new password");
        return;
      }

      fetch("/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, new_password: newPassword })
      })
      .then(res => res.json())
      .then(data => {
        if (data.message) {
          document.getElementById("auth-status").textContent = data.message + " You can now login.";
          document.getElementById("auth-status").classList.remove("d-none");
          // Switch back to login tab
          document.getElementById("show-login").click();
        } else {
          alert(data.error);
        }
      });
    };
  });