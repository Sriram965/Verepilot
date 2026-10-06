from __future__ import annotations

import http.cookiejar
import urllib.error
import urllib.request
from urllib.parse import urlencode, urljoin, urlparse


class _NoRedirectHandler(
    urllib.request.HTTPRedirectHandler
):
    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        return None


class ShopClient:
    def __init__(
        self,
        base_url: str,
    ):
        parsed = urlparse(base_url)

        if (
            parsed.scheme != "http"
            or parsed.hostname
            not in {
                "127.0.0.1",
                "localhost",
            }
        ):
            raise ValueError(
                "ShopClient accepts only "
                "loopback HTTP origins"
            )

        self.base_url = base_url.rstrip("/")

        self.cookie_jar = (
            http.cookiejar.CookieJar()
        )

        self.opener = (
            urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(
                    self.cookie_jar
                ),
                _NoRedirectHandler(),
            )
        )

        self.last_url = ""

    def request(
        self,
        path_or_url: str,
        method: str = "GET",
        data: dict[str, str] | None = None,
    ) -> tuple[int, str]:

        url = urljoin(
            self.base_url + "/",
            path_or_url,
        )

        if (
            urlparse(url).netloc
            != urlparse(
                self.base_url
            ).netloc
        ):
            raise ValueError(
                "cross-origin requests "
                "are forbidden"
            )

        body = None
        headers: dict[str, str] = {}

        if data is not None:
            body = urlencode(
                data
            ).encode("utf-8")

            headers[
                "Content-Type"
            ] = (
                "application/"
                "x-www-form-urlencoded"
            )

        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with self.opener.open(
                request,
                timeout=5,
            ) as response:

                payload = (
                    response.read()
                    .decode("utf-8")
                )

                self.last_url = (
                    response.geturl()
                )

                return (
                    response.status,
                    payload,
                )

        except urllib.error.HTTPError as exc:
            payload = (
                exc.read()
                .decode("utf-8")
            )

            self.last_url = (
                exc.geturl()
            )

            return (
                exc.code,
                payload,
            )

    def get(
        self,
        path: str,
    ) -> tuple[int, str]:
        return self.request(
            path,
            "GET",
        )

    def post(
        self,
        path: str,
        data: dict[str, str],
    ) -> tuple[int, str]:

        return self.request(
            path,
            "POST",
            data,
        )

    def session_id(self) -> str:
        for cookie in self.cookie_jar:
            if cookie.name == "vp_session":
                return cookie.value

        raise RuntimeError(
            "session cookie not established"
        )
