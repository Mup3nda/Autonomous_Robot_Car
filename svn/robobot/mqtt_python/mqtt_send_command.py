#!/usr/bin/env python3
"""
Simple utility to send MQTT commands to the robot
Usage:
    python3 mqtt_send_command.py calibrate_line    # Calibrate line sensor
    python3 mqtt_send_command.py drive 0.2 0.0     # Drive forward
    python3 mqtt_send_command.py custom "licw 50"  # Send custom command
"""

import sys
import argparse
from paho.mqtt import client as mqtt_client
import time

MQTT_HOST = 'localhost'
MQTT_PORT = 1883
MQTT_TIMEOUT = 1.0

# Common topics
TOPIC_CMD_T0 = "robobot/cmd/T0/"
TOPIC_DRIVE = "robobot/drive/"

# Preset commands
COMMANDS = {
    'calibrate_line': {
        'description': 'Calibrate line sensor white level',
        'topic': TOPIC_CMD_T0,
        'message': 'licw 100',
        'post_command': [
            {'delay': 0.3, 'topic': TOPIC_CMD_T0, 'message': 'eew', 'description': 'Save to EEPROM'}
        ]
    },
    'calibrate_black': {
        'description': 'Calibrate line sensor black level',
        'topic': TOPIC_CMD_T0,
        'message': 'licb 100',
    },
    'get_line_raw': {
        'description': 'Get raw line sensor values',
        'topic': TOPIC_CMD_T0,
        'message': 'liv',
    },
    'get_line_white': {
        'description': 'Get line sensor white level',
        'topic': TOPIC_CMD_T0,
        'message': 'liwi',
    },
    'get_line_black': {
        'description': 'Get line sensor black level',
        'topic': TOPIC_CMD_T0,
        'message': 'libi',
    },
    'get_line_pos': {
        'description': 'Get line position and validity',
        'topic': TOPIC_CMD_T0,
        'message': 'lip',
    },
    'get_line_status': {
        'description': 'Get all line sensor settings',
        'topic': TOPIC_CMD_T0,
        'message': 'lis',
    },
    'stop': {
        'description': 'Stop the robot',
        'topic': TOPIC_CMD_T0,
        'message': 'stop',
    },
    'help': {
        'description': 'Show all available commands on Teensy',
        'topic': TOPIC_CMD_T0,
        'message': 'help',
    },
}

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker"""
    if rc == 0:
        print("✓ Connected to MQTT broker")
    else:
        print(f"✗ Connection failed with code {rc}")

def on_disconnect(client, userdata, rc):
    """Callback for when the client disconnects"""
    if rc != 0:
        print(f"✗ Unexpected disconnection: {rc}")

def on_publish(client, userdata, mid):
    """Callback for when a message is published"""
    pass

def send_mqtt_command(host, topic, message):
    """Send a single MQTT message"""
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, client_id="mqtt-send-cmd")
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    
    try:
        client.connect(host, MQTT_PORT, keepalive=60)
        client.loop_start()
        
        # Give it a moment to connect
        time.sleep(0.2)
        
        # Publish the message
        result = client.publish(topic, message, qos=1)
        if result[0] == 0:
            print(f"✓ Sent: '{message}' to topic '{topic}'")
            time.sleep(0.5)  # Wait for response
        else:
            print(f"✗ Failed to send message (error code: {result[0]})")
        
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        print(f"✗ Error: {e}")

def list_commands():
    """List all available preset commands"""
    print("\nAvailable preset commands:")
    print("=" * 60)
    for cmd, details in COMMANDS.items():
        print(f"  {cmd:20} - {details['description']}")
        print(f"    → '{details['message']}'")
        if 'post_command' in details:
            for post_cmd in details['post_command']:
                print(f"    → (after {post_cmd['delay']}s) '{post_cmd['message']}'")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='Send MQTT commands to the robot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 mqtt_send_command.py calibrate_line
  python3 mqtt_send_command.py get_line_pos
  python3 mqtt_send_command.py custom "licw 50"
  python3 mqtt_send_command.py custom -t "robobot/cmd/T0/" "your_command"
  python3 mqtt_send_command.py --list
        """
    )
    
    parser.add_argument('command', nargs='?', help='Preset command or "custom"')
    parser.add_argument('message', nargs='?', help='Message to send (for custom commands)')
    parser.add_argument('-H', '--host', default=MQTT_HOST, help=f'MQTT host (default: {MQTT_HOST})')
    parser.add_argument('-t', '--topic', default=TOPIC_CMD_T0, help=f'MQTT topic (default: {TOPIC_CMD_T0})')
    parser.add_argument('-l', '--list', action='store_true', help='List all preset commands')
    
    args = parser.parse_args()
    
    # Show list of commands if requested
    if args.list:
        list_commands()
        return
    
    # Need a command
    if not args.command:
        parser.print_help()
        list_commands()
        return
    
    # Handle preset commands
    if args.command in COMMANDS:
        cmd_info = COMMANDS[args.command]
        print(f"\n{cmd_info['description']}")
        send_mqtt_command(args.host, cmd_info['topic'], cmd_info['message'])
        
        # Handle post commands (like saving to EEPROM)
        if 'post_command' in cmd_info:
            for post_cmd in cmd_info['post_command']:
                print(f"\n(waiting {post_cmd['delay']}s) {post_cmd['description']}")
                time.sleep(post_cmd['delay'])
                send_mqtt_command(args.host, post_cmd['topic'], post_cmd['message'])
    
    # Handle custom commands
    elif args.command == 'custom':
        if not args.message:
            print("✗ Error: custom command requires a message argument")
            print(f"  Usage: python3 mqtt_send_command.py custom \"your_command\"")
            return
        send_mqtt_command(args.host, args.topic, args.message)
    
    else:
        print(f"✗ Unknown command: {args.command}")
        list_commands()

if __name__ == '__main__':
    main()
