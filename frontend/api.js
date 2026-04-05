function getApiBaseUrl() {
    const { protocol, hostname, port } = window.location;

    if (protocol === "file:" || port === "5500") {
        return `http://${hostname || "127.0.0.1"}:8000`;
    }

    return "";
}

function buildApiUrl(path) {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    return `${getApiBaseUrl()}${normalizedPath}`;
}

function saveSession(authPayload) {
    if (!authPayload || !authPayload.access_token || !authPayload.user) {
        return;
    }

    localStorage.setItem("access_token", authPayload.access_token);
    localStorage.setItem("ktu_user", JSON.stringify(authPayload.user));
}

function getStoredUser() {
    const rawUser = localStorage.getItem("ktu_user");

    if (!rawUser) {
        return null;
    }

    try {
        return JSON.parse(rawUser);
    } catch (error) {
        localStorage.removeItem("ktu_user");
        return null;
    }
}

function clearSession() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("ktu_user");
}

function getStoredToken() {
    return localStorage.getItem("access_token");
}
