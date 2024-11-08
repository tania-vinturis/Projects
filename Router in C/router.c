#include <arpa/inet.h> /* ntoh, hton and inet_ functions */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "lib.h"
#include "protocols.h"
#include "queue.h"

#define ETHERTYPE_IP 0x0800
#define ETHERTYPE_ARP 0x0806
#define ICMP_TYPE_TIME_OUT 11
#define ICMP_DESTINATION_UNREACHABLE 3
#define MAX_PACKET_LEN 1600
#define IPROTO_ICMP 1

int rtable_len ;
struct route_table_entry *rtable;

int arp_table_len;
struct arp_table_entry *arp_table;
	
struct route_table_entry *get_best_route_recursive(int left, int right, uint32_t dest_ip, struct route_table_entry *best_route) {
    if (left > right) {
        return best_route;
    }

    int mid = left + (right - left) / 2;
    
    if ((dest_ip & rtable[mid].mask) == (rtable[mid].prefix & rtable[mid].mask)) {
        if (best_route == NULL || ntohl(rtable[mid].mask) > ntohl(best_route->mask)) {
            best_route = &rtable[mid];
        }
        return get_best_route_recursive(left, mid - 1, dest_ip, best_route);
    } else if (ntohl(dest_ip & rtable[mid].mask) > ntohl(rtable[mid].prefix & rtable[mid].mask)) {
        // Caut in stanga
        return get_best_route_recursive(left, mid - 1, dest_ip, best_route);
    } else {
        // Caut in dreapta
        return get_best_route_recursive(mid + 1, right, dest_ip, best_route);
    }
}

struct route_table_entry *get_best_route(uint32_t dest_ip) {
    return get_best_route_recursive(0, rtable_len - 1, dest_ip, NULL);
}

struct arp_table_entry *get_arp_entry(uint32_t ip) {
    for (int i = 0; i < arp_table_len; i++) {
        if ((arp_table + i)->ip == ip) 
            return (arp_table + i);
    }
    return NULL;
}

// Functia pt qsort
int cmp(const void *first, const void *second) {
    const struct route_table_entry *a = (const struct route_table_entry *)first;
    const struct route_table_entry *b = (const struct route_table_entry *)second;

	int a_network = ntohl(a->mask & a->prefix);
	int b_network = ntohl(b->mask & b->prefix);
    // ordonare dupa masca
    if (a_network != b_network) {
		return  b_network > a_network;
    } // ordonare dupa prefix
    return ntohl(b->mask) > ntohl(a->mask);
}


struct iphdr* function_ip_header(char *icmp_packet, int version, int ihl, int tos, int total_len, int id, int frag_off,
                                int ttl, int protocol, int saddr, int daddr, int check, int interface) {

    struct iphdr *new_ip_hdr = (struct iphdr *)(icmp_packet + sizeof(struct ether_header));
    new_ip_hdr->version = version;
    new_ip_hdr->ihl = ihl;
    new_ip_hdr->tos = tos;
    new_ip_hdr->tot_len = total_len;
    new_ip_hdr->id = id;
    new_ip_hdr->frag_off = frag_off;
    new_ip_hdr->ttl = ttl;
    new_ip_hdr->protocol = protocol;
    new_ip_hdr->check = check;
    new_ip_hdr->saddr = saddr;
    new_ip_hdr->daddr = daddr;
    return new_ip_hdr;
}

// Funcție pentru construirea și trimiterea unui pachet ICMP
void helper_icmp(char *buf, int interface, struct ether_header *eth_hdr, struct iphdr *ip_hdr, size_t len, uint8_t type) {
    size_t icmp_msg_len = sizeof(struct icmphdr) + sizeof(struct iphdr) + 8; // 8 octeți din payload
    size_t total_len = sizeof(struct ether_header) + sizeof(struct iphdr) + icmp_msg_len;
    char *icmp_packet = malloc(total_len);
    
    // Construiesc antet ETH nou
    struct ether_header *new_eth_hdr = (struct ether_header *)icmp_packet;
    memcpy(new_eth_hdr->ether_dhost, eth_hdr->ether_shost, 6);
    memcpy(new_eth_hdr->ether_shost, eth_hdr->ether_dhost, 6);
    new_eth_hdr->ether_type = htons(ETHERTYPE_IP);
	struct iphdr *new_ip_hdr = function_ip_header(icmp_packet, 4, 5, 0, htons(sizeof(struct iphdr) + icmp_msg_len), 0, 0, 64, IPROTO_ICMP, inet_addr(get_interface_ip(interface)),ip_hdr->saddr, 0, interface);
    // Construiesc antet ICMP
    struct icmphdr *icmp_hdr = (struct icmphdr *)(icmp_packet + sizeof(struct ether_header) + sizeof(struct iphdr));
    icmp_hdr->type = type;
    icmp_hdr->code = 0;
    icmp_hdr->checksum = 0;
    icmp_hdr->un.echo.id = 0;
    icmp_hdr->un.echo.sequence = 0;
    
    // Copiez antet IP + primele 8 octeti din payload
    memcpy(icmp_hdr + 1, ip_hdr, sizeof(struct iphdr) + 8);
    
    // Calculez checksum pentru antetul IP nou
    new_ip_hdr->check = checksum((uint16_t *)new_ip_hdr, sizeof(struct iphdr));
    
    // Calculez checksum pentru ICMP
    icmp_hdr->checksum = checksum((uint16_t *)icmp_hdr, icmp_msg_len);
    
    // Trimit pachetul construit
    send_to_link(interface, icmp_packet, total_len);
    free(icmp_packet);
}

void send_icmp_timeout(char *buf, int interface, struct ether_header *eth_hdr, struct iphdr *ip_hdr, size_t len) {
    helper_icmp(buf, interface, eth_hdr, ip_hdr, len, ICMP_TYPE_TIME_OUT);
}

void send_icmp_destination_unreachable(char *buf, int interface, struct ether_header *eth_hdr, struct iphdr *ip_hdr, size_t len) {
    helper_icmp(buf, interface, eth_hdr, ip_hdr, len, ICMP_DESTINATION_UNREACHABLE);
}

void swap_mac_addresses(uint8_t *src_mac, uint8_t *dst_mac) {
    uint8_t temp[6];
    memcpy(temp, src_mac, 6);
    memcpy(src_mac, dst_mac, 6);
    memcpy(dst_mac, temp, 6);
}

void icmp_echo_reply(char* buf, int interface, struct ether_header *eth_hdr, struct iphdr *ip_hdr, size_t len) {
    // Inversez sursa/destinatia
    swap_mac_addresses(eth_hdr->ether_shost, eth_hdr->ether_dhost);
    // Trimit pachetul icmp echo reply
    helper_icmp(buf, interface, eth_hdr, ip_hdr, len, 0);
}

int main(int argc, char *argv[])
{
	char buf[MAX_PACKET_LEN];
    
	// Do not modify this line
	init(argc - 2, argv + 2);

	rtable = malloc(sizeof(struct route_table_entry)*100000);
	rtable_len = read_rtable(argv[1], rtable);
	arp_table = malloc(sizeof(struct route_table_entry)*100000);
	arp_table_len = parse_arp_table("arp_table.txt", arp_table);
    
	qsort(rtable, rtable_len, sizeof(struct route_table_entry), cmp);

    uint8_t mac_address[6];
	while (1) {

		int interface;
		size_t len;

		interface = recv_from_any_link(buf, &len);
		DIE(interface < 0, "recv_from_any_links");

		struct ether_header *eth_hdr = (struct ether_header *) buf;
		struct iphdr *ip_hdr = (struct iphdr *)(buf + sizeof(struct ether_header));
		
		//uint8_t *mac = 0;
		uint8_t broadcast[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

		int is_ok = inet_addr(get_interface_ip(interface)) == ip_hdr->daddr;
		//get_interface_mac(rtable->interface, mac);
		// if(eth_hdr->ether_dhost != mac && eth_hdr->ether_dhost != broadcast){
		// 	printf("Trimiti catre cine??");
		// 	continue;
		// }

		// if (memcmp(eth_hdr->ether_dhost, mac, 6) != 0 && memcmp(eth_hdr->ether_dhost, broadcast, 6) != 0) {
		// 	printf("Trimiti catre cine??");
		// 	continue;
		// 	}

		if(eth_hdr->ether_type == ntohs(ETHERTYPE_IP)){
				uint16_t checksum_anterior = ip_hdr->check;
                ip_hdr->check = 0;
                if (checksum_anterior != htons(checksum((uint16_t *)ip_hdr, sizeof(struct iphdr)))) {
					printf("Failed checksum!");
					memset(buf, 0, sizeof(buf));
                    continue;
                }
				struct route_table_entry *best_route = get_best_route(ip_hdr->daddr);
				if(best_route != NULL){
					
						if(ip_hdr->ttl > 1){
							if(is_ok){
								//icmp echo reply
                                icmp_echo_reply(buf, interface, eth_hdr, ip_hdr, len);
								continue;
							}
							else{
								uint16_t old_ttl = ip_hdr->ttl;
								ip_hdr->ttl--;
								ip_hdr->check = ~(~checksum_anterior +  ~((uint16_t)old_ttl) 
															+ (uint16_t)ip_hdr->ttl) - 1;
								struct arp_table_entry *next_destionation = get_arp_entry(best_route->next_hop);
								for (int i = 0; i < 6; i++) {
										eth_hdr->ether_dhost[i] = next_destionation->mac[i];
									}
								
								send_to_link(best_route->interface, buf, len);
							}
							
						}
						else{
							//icmp timeout
                            send_icmp_timeout(buf, interface, eth_hdr, ip_hdr, len);
							printf("icmp time out");
							continue;
						}
				}
				else{
					//icmp dest unreachable
					send_icmp_destination_unreachable(buf, interface, eth_hdr, ip_hdr, len);
					printf("Destionation unreachable");
					continue;
				}

		}
		/* Note that packets received are in network order,
		any header field which has more than 1 byte will need to be conerted to
		host order. For example, ntohs(eth_hdr->ether_type). The oposite is needed when
		sending a packet on the link, */
		continue;
	}
	
}
