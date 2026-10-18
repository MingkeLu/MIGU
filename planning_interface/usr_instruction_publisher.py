#!/usr/bin/env python3
"""
Publisher for user instructions to the /usr_instruction ROS topic.
This is a simulation/testing tool to send text instructions to the planner.

Usage:
    python usr_instruction_publisher.py --text "put the red cube on the green cube"
    python usr_instruction_publisher.py --interactive  # Interactive mode to type instruction
"""

from __future__ import print_function

import argparse
import time
import rospy
from std_msgs.msg import String


def publish_instruction(publisher, instruction_text):
    """
    Publish a user instruction to the ROS topic.
    
    Args:
        publisher: ROS publisher for String messages
        instruction_text: The instruction text to send
    """
    msg = String()
    msg.data = instruction_text
    publisher.publish(msg)
    print("Published instruction: '{}'".format(instruction_text))
    rospy.sleep(0.1)  # Small delay to ensure message is sent


def main():
    """
    Publisher for user instructions targeting the planner.
    Messages are published to the /usr_instruction ROS topic.
    
    Command-line parameters:
    - --text (str): The instruction text to send (if not provided, uses interactive mode)
    - --topic (str, default: '/usr_instruction'): ROS topic name for publishing instructions
    - --interactive (flag): Interactive mode to type instruction from stdin
    """
    parser = argparse.ArgumentParser(description="User instruction publisher (ROS-based)")
    parser.add_argument("--text", type=str, default=None,
                        help="The instruction text to send")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode to type instruction from stdin")
    parser.add_argument("--topic", type=str, default="/usr_instruction",
                        help="ROS topic name for publishing instructions")
    
    args = parser.parse_args()
    
    # Initialize ROS node
    try:
        rospy.init_node('usr_instruction_publisher', anonymous=True)
    except rospy.ROSException as e:
        print("ROS node may already be initialized: {}".format(e))
    
    # Create publisher
    pub = rospy.Publisher(args.topic, String, queue_size=10)
    
    # Wait for subscribers to connect
    print("Waiting for subscribers to connect...")
    rospy.sleep(0.5)
    
    # Get instruction text
    instruction_text = None
    if args.interactive:
        # Interactive mode: read from stdin
        print("Enter instruction (press Enter when done):")
        instruction_text = input().strip()
    elif args.text:
        # Use provided text
        instruction_text = args.text
    else:
        parser.error("provide --text or --interactive")
    
    if not instruction_text:
        print("Error: Empty instruction text. Exiting.")
        return
    
    # Publish the instruction
    print("Publishing instruction to topic: {}".format(args.topic))
    publish_instruction(pub, instruction_text)
    
    # Small delay to ensure message is sent
    rospy.sleep(0.2)
    
    print("Instruction published successfully. Exiting.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()

