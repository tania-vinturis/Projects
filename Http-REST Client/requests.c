#include <stdlib.h>     /* exit, atoi, malloc, free */
#include <stdio.h>
#include <unistd.h>     /* read, write, close */
#include <string.h>     /* memcpy, memset */
#include <sys/socket.h> /* socket, connect */
#include <netinet/in.h> /* struct sockaddr_in, struct sockaddr */
#include <netdb.h>      /* struct hostent, gethostbyname */
#include <arpa/inet.h>
#include "helpers.h"
#include "requests.h"

char *compute_get_request(char *host, char *url, char *query_params,
                          char **cookies, int cookies_count, const char *token) {
    char *message = calloc(BUFLEN, sizeof(char));
    char *line = calloc(LINELEN, sizeof(char));

    if (query_params != NULL) {
        sprintf(line, "GET %s?%s HTTP/1.1", url, query_params);
    } else {
        sprintf(line, "GET %s HTTP/1.1", url);
    }
    compute_message(message, line);

    sprintf(line, "Host: %s", host);
    compute_message(message, line);

    if (token != NULL) {
        sprintf(line, "Authorization: Bearer %s", token);
        compute_message(message, line);
    }

    if (cookies != NULL) {
        char *at = line;
        size_t remaining_size = LINELEN;

        at += snprintf(at, remaining_size, "Cookie:");
        remaining_size -= strlen("Cookie:");

        for (int i = 0; i < cookies_count; i++) {
            size_t len = snprintf(at, remaining_size, " %s%s", cookies[i], (i < cookies_count - 1) ? ";" : "");
            at += len;
            remaining_size -= len;
        }
        compute_message(message, line);
    }


    compute_message(message, "");
    free(line);
    return message;
}

char *compute_post_request(char *host, char *url, char *content_type, char **body_data,
                           int body_data_fields_count, char **cookies, int cookies_count, const char *token) {
    char *message = calloc(BUFLEN, sizeof(char));
    char *line = calloc(LINELEN, sizeof(char));
    char *body_data_buffer = calloc(LINELEN, sizeof(char));

    sprintf(line, "POST %s HTTP/1.1", url);
    compute_message(message, line);

    sprintf(line, "Host: %s", host);
    compute_message(message, line);

    if (token != NULL) {
        sprintf(line, "Authorization: Bearer %s", token);
        compute_message(message, line);
    }

    sprintf(line, "Content-Type: %s", content_type);
    compute_message(message, line);

    int data_len = 0;
    for (int i = 0; i < body_data_fields_count; i++) {
        data_len += strlen(body_data[i]);
        if (i < body_data_fields_count - 1) {
            data_len++;
        }
    }

    sprintf(line, "Content-Length: %d", data_len);
    compute_message(message, line);


    if (cookies != NULL) {
        char *at = line;
        size_t remaining_size = LINELEN;

        at += snprintf(at, remaining_size, "Cookie:");
        remaining_size -= strlen("Cookie:");

        for (int i = 0; i < cookies_count; i++) {
            size_t len = snprintf(at, remaining_size, " %s%s", cookies[i], (i < cookies_count - 1) ? ";" : "");
            at += len;
            remaining_size -= len;
        }

        compute_message(message, line);
    }

    compute_message(message, "");

    for (int i = 0; i < body_data_fields_count; i++) {
        sprintf(body_data_buffer, "%s", body_data[i]);
        compute_message(message, body_data_buffer);
    }

    free(line);
    free(body_data_buffer);
    return message;
}

char *compute_delete_request(char *host, char *url, char **cookies, int cookies_count, const char *token) {
    char *message = calloc(BUFLEN, sizeof(char));
    char *line = calloc(LINELEN, sizeof(char));

    sprintf(line, "DELETE %s HTTP/1.1", url);
    compute_message(message, line);

    sprintf(line, "Host: %s", host);
    compute_message(message, line);

    if (token != NULL) {
        sprintf(line, "Authorization: Bearer %s", token);
        compute_message(message, line);
    }

    if (cookies != NULL) {
        char *at = line;
        size_t remaining_size = LINELEN;

        at += snprintf(at, remaining_size, "Cookie:");
        remaining_size -= strlen("Cookie:");

        for (int i = 0; i < cookies_count; i++) {
            size_t len = snprintf(at, remaining_size, " %s%s", cookies[i], (i < cookies_count - 1) ? ";" : "");
            at += len;
            remaining_size -= len;
        }

        compute_message(message, line);
    }


    compute_message(message, "");
    free(line);
    return message;
}