# Router Dataplane Implementation

## Project Overview

This project implements the **dataplane** component of a router in C, handling packet forwarding based on a static routing table. The dataplane processes each packet by finding the appropriate next hop and handling the packet according to routing rules, including ICMP protocol responses. The routing table is provided as an input file and remains static throughout execution.

The **control plane** is not included in this project, meaning no dynamic routing protocols (RIP, OSPF, BGP) are implemented. Instead, the focus is strictly on the dataplane, where the router uses pre-loaded routing table entries to make forwarding decisions.

## Key Functionalities

1. **Routing Process**:
   - Implements packet processing and forwarding based on a static routing table, including handling TTL, ICMP responses, and checksums.
   
2. **Longest Prefix Match (LPM)**:
   - Utilizes binary search for efficient LPM using `qsort` to sort the routing table by mask and prefix, optimizing routing lookups.
   
3. **ICMP Protocol Support**:
   - Handles ICMP responses for Echo Reply, Timeout, and Destination Unreachable messages.

## Detailed Description of Implementation

### Main Function (main.c)

The main function initializes the router, allocates resources, and continuously processes packets in a loop:
- **Initialization**:
  - Sets up the `route_table` and loads entries.
  - Utilizes a static ARP table for resolving IP to MAC addresses in this implementation.
  - Sorts the `route_table` using `qsort` for binary search efficiency.
  
- **Packet Processing Loop**:
  - Each incoming packet is processed as follows:
    - **Type Check**: Verifies that the packet is of IP type.
    - **Checksum Calculation**: Validates checksum using a provided function.
    - **Route Lookup**: Calculates the best route using the `get_best_route` function.
    - **TTL and ICMP Handling**:
      - If a route is found:
        - **TTL Check**: If `TTL > 1`, proceed with routing; otherwise, trigger an ICMP Timeout.
        - **ICMP Echo Reply**: If the packet is addressed to the router, an Echo Reply is sent.
        - **Packet Forwarding**: 
          - Decrement TTL, recalculate checksum, and find the next hop using `get_arp_entry`.
          - Update the MAC address in `dhost` to the destination’s MAC and forward the packet via `send_to_link`.
      - If no route is found, trigger an ICMP Destination Unreachable.

### Helper Functions

- **get_best_route_recursive()**: Implements Longest Prefix Match (LPM) using binary search.
  
- **get_best_route()**: Wrapper function that calls `get_best_route_recursive()` with the appropriate parameters.

- **get_arp_entry()**: Retrieves an ARP entry for a given IP address from the ARP table.

- **cmp()**: Comparison function for `qsort`, sorting `route_table` entries by mask and prefix for efficient LPM.

- **function_ip_header()**: Updates fields of the IP header, avoiding code repetition.

- **helper_icmp()**: Builds and sends an ICMP packet. Steps include:
  - Constructing a new Ethernet header.
  - Building an ICMP header.
  - Copying IP and the first 8 bytes of the payload.
  - Calculating checksums for both IP and ICMP headers.
  - Sending the packet.

- **send_icmp_timeout()**: Calls `helper_icmp` with parameters specific to an ICMP Timeout response.

- **icmp_echo_reply()**: Handles ICMP Echo Reply by:
  - Swapping source and destination MAC addresses.
  - Calling `helper_icmp` with Echo Reply parameters.

## File Structure

- **main.c**: Initializes the router, processes packets in a loop, and manages ICMP handling.
- **router.c**: Implements core dataplane functionality, including route lookup and packet forwarding.
- **routing_table.c**: Loads and manages the static routing table from an input file.
- **arp_table.c**: Manages ARP table lookups for IP to MAC resolution.
- **icmp.c**: Contains functions for handling ICMP packets, including Timeout and Destination Unreachable messages.
- **packet.c**: Defines packet structures and functions to manipulate packet headers.
- **interface.c**: Manages the router’s interfaces and packet transmission.

## How to Run

1. **Compile the Project**:
   ```bash
   gcc -o router main.c router.c routing_table.c arp_table.c icmp.c packet.c interface.c
