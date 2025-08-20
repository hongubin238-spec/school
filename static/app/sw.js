// very simple service worker
self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener("fetch", (event) => {
  // 그냥 네트워크로 바로 요청 전달
  event.respondWith(fetch(event.request));
});
