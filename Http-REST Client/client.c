#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include "helpers.h"
#include "requests.h"
#include "parson.h"

#define API_HOST "34.246.184.49:8080"
#define MAX_LEN 1000

char authCookie[MAX_LEN];
char libraryToken[MAX_LEN];

void performUserOperation(const char *operation);

void sendRequestAndHandleResponse(char *message, int (responseHandler)(char *));

void performLibraryOperation(const char *operation, const char *additionalPath);

void readStringInput(const char *prompt, char *buffer, size_t bufferSize);

int handleStandardResponse(const char *response);

int extractTokenAndHandleResponse(const char *response);

int extractCookieAndHandleResponse(const char *response);

char *extractResponseBody(const char *response);

void clearAuthDetails();

void readBookDetails(JSON_Object *root_object);

int handleGetBooksResponse(const char *response);


int main(int argc, char *argv[]) {
    char command[50];
    int continueRunning = 1;

    while (continueRunning) {

        if (scanf("%s", command) != 1) {
            printf("Error reading command.\n");
            continue;
        }

        if (strcmp(command, "exit") == 0) {
            continueRunning = 0;
        } else {
            performUserOperation(command);
        }
    }

    return 0;
}

void performUserOperation(const char *operation) {
    if (strcmp(operation, "register") == 0 || strcmp(operation, "login") == 0) {
        char username[50], password[50];
        readStringInput("username=", username, sizeof(username));
        readStringInput("password=", password, sizeof(password));

        JSON_Value *root_value = json_value_init_object();
        JSON_Object *root_object = json_value_get_object(root_value);
        json_object_set_string(root_object, "username", username);
        json_object_set_string(root_object, "password", password);
        char *serialized_string = json_serialize_to_string_pretty(root_value);

        char *body_data[] = {serialized_string};
        char *message = compute_post_request(API_HOST, strcmp(operation, "register") == 0 ? "/api/v1/tema/auth/register"
                                                                                          : "/api/v1/tema/auth/login",
                                             "application/json", body_data, 1, NULL, 0, NULL);

        sendRequestAndHandleResponse(message,
                                     (int (*)(char *)) (strcmp(operation, "login") == 0 ? extractCookieAndHandleResponse
                                                                                        : handleStandardResponse));

        json_free_serialized_string(serialized_string);
        json_value_free(root_value);

    } else if (strcmp(operation, "logout") == 0) {
        char *message = compute_get_request(API_HOST, "/api/v1/tema/auth/logout", NULL, (char *[]) {authCookie}, 1,
                                            NULL);
        sendRequestAndHandleResponse(message, (int (*)(char *)) handleStandardResponse);
        clearAuthDetails();
    } else {
        performLibraryOperation(operation, NULL);
    }
}

void performLibraryOperation(const char *operation, const char *additionalPath) {
    char url[100] = "/api/v1/tema/library/";
    char *message;
    if (strcmp(operation, "enter_library") == 0) {
        strcat(url, "access");
        message = compute_get_request(API_HOST, url, NULL, (char *[]) {authCookie}, 1, libraryToken);
        sendRequestAndHandleResponse(message, (int (*)(char *)) (strcmp(operation, "enter_library") == 0
                                                                 ? extractTokenAndHandleResponse
                                                                 : handleStandardResponse));
    } else if (strcmp(operation, "add_book") == 0) {
        strcat(url, "/books");
        JSON_Value *root_value = json_value_init_object();
        JSON_Object *root_object = json_value_get_object(root_value);
        readBookDetails(root_object);
        char *serialized_string = json_serialize_to_string_pretty(root_value);

        char *body_data[] = {serialized_string};
        message = compute_post_request(API_HOST, url, "application/json", body_data, 1, (char *[]) {authCookie}, 1,
                                       libraryToken);
        sendRequestAndHandleResponse(message, (int (*)(char *)) handleStandardResponse);
        json_free_serialized_string(serialized_string);
        json_value_free(root_value);
    } else if (strcmp(operation, "delete_book") == 0) {
        int bookId;
        char bookid[100];
        strcat(url, "books/");
        printf("id=");
        scanf("%d", &bookId);
        sprintf(bookid, "%d", bookId);
        strcat(url, bookid);
        message = compute_delete_request(API_HOST, url, (char *[]) {authCookie}, 1, libraryToken);
        sendRequestAndHandleResponse(message, (int (*)(char *)) handleGetBooksResponse);


    } else if (strcmp(operation, "get_books") == 0) {
        strcat(url, "books");
        message = compute_get_request(API_HOST, url, NULL, (char *[]) {authCookie}, 1, libraryToken);
        sendRequestAndHandleResponse(message, (int (*)(char *)) handleGetBooksResponse);
    } else if (strcmp(operation, "get_book") == 0) {
        int bookId;
        char bookid[100];
        strcat(url, "books/");
        printf("id=");
        scanf("%d", &bookId);
        sprintf(bookid, "%d", bookId);
        strcat(url, bookid);
        message = compute_get_request(API_HOST, url, NULL, (char *[]) {authCookie}, 1, libraryToken);
        sendRequestAndHandleResponse(message, (int (*)(char *)) handleGetBooksResponse);
    }
}


void readBookDetails(JSON_Object *root_object) {
    
    char title[150];
    char author[150];
    char genre[50];
    char publisher[150];
    char pages[50];
    fgets(title, sizeof(title), stdin);

    printf("title= ");
    fgets(title, sizeof(title), stdin);
    title[strcspn(title, "\n")] = 0;

    printf("author= ");
    fgets(author, sizeof(author), stdin);
    author[strcspn(author, "\n")] = 0;

    printf("genre= ");
    fgets(genre, sizeof(genre), stdin);
    genre[strcspn(genre, "\n")] = 0;

    printf("publisher= ");
    fgets(publisher, sizeof(publisher), stdin);
    publisher[strcspn(publisher, "\n")] = 0;

    printf("page_count= ");
    fgets(pages, sizeof(pages), stdin);
    pages[strcspn(pages, "\n")] = 0;

    json_object_set_string(root_object, "title", title);
    json_object_set_string(root_object, "author", author);
    json_object_set_string(root_object, "genre", genre);
    json_object_set_string(root_object, "publisher", publisher);
    json_object_set_number(root_object, "page_count", atoi(pages));
}

void sendRequestAndHandleResponse(char *message, int (responseHandler)(char *)) {
    int sockfd = open_connection("34.246.184.49", 8080, AF_INET, SOCK_STREAM, 0);
    send_to_server(sockfd, message);
    char *response = receive_from_server(sockfd);
    if (!responseHandler(response)) {
        printf("%s\n", extractResponseBody(response));
    } else {
        printf("Success!\n");
    }
    close(sockfd);
}

int handleGetBooksResponse(const char *response) {
    if (!handleStandardResponse(response)) {
        printf("Failed to fetch books.\n");
        return 0;
    }
    const char *responseBody = extractResponseBody(response);
    printf("%s\n", responseBody);

    return 1;
}

void readStringInput(const char *prompt, char *buffer, size_t bufferSize) {
    printf("%s", prompt);
    scanf("%s", buffer);
}

int handleStandardResponse(const char *response) {
    return strstr(response, "HTTP/1.1 20") != NULL;
}

int extractTokenAndHandleResponse(const char *response) {
    if (handleStandardResponse(response)) {
        const char *responseBody = extractResponseBody(response);

        JSON_Value *root_value = json_parse_string(responseBody);
        if (root_value == NULL) {
            printf("Error parsing JSON.\n");
            return 0;
        }

        JSON_Object *root_object = json_value_get_object(root_value);
        if (root_object == NULL) {
            json_value_free(root_value);
            printf("Error getting JSON object.\n");
            return 0;
        }

        const char *token = json_object_get_string(root_object, "token");
        if (token == NULL) {
            json_value_free(root_value);
            printf("Token not found in JSON.\n");
            return 0;
        }

        strncpy(libraryToken, token, MAX_LEN - 1);
        libraryToken[MAX_LEN - 1] = '\0';

        json_value_free(root_value);
        return 1;
    }
    return 0;
}

int extractCookieAndHandleResponse(const char *response) {
    if (!handleStandardResponse(response)) {
        return 0;
    }

    const char *set_cookie_header = "Set-Cookie: ";
    const char *cookie_start = strstr(response, set_cookie_header);
    if (!cookie_start) {
        return 0;
    }

    cookie_start += strlen(set_cookie_header);
    const char *cookie_end = strchr(cookie_start, ';');
    if (cookie_end == NULL) {
        cookie_end = cookie_start + strlen(cookie_start);
    }

    size_t cookie_length = cookie_end - cookie_start;
    if (cookie_length >= MAX_LEN) {
        cookie_length = MAX_LEN - 1;
    }
    strncpy(authCookie, cookie_start, cookie_length);
    authCookie[cookie_length] = '\0';

    return 1;
}


char *extractResponseBody(const char *response) {
    char *body = strstr(response, "\r\n\r\n");
    return body ? body + 4 : "";
}

void clearAuthDetails() {
    memset(authCookie, 0, sizeof(authCookie));
    memset(libraryToken, 0, sizeof(libraryToken));
}