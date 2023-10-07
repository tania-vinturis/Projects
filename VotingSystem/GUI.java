package com.company;

import jdk.swing.interop.SwingInterOpUtils;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GUI {
    JFrame frame;
    GUI(){
        frame = new JFrame("Online Voting System");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300,300);
        JButton button1 = new JButton("Choose the candidate you want to vote");
        JButton button2 = new JButton("Thank you for your time!");
        JButton button3 = new JButton("CANDIDATE3");
        JButton button4 = new JButton("CANDIDATE1");
        JButton button5 = new JButton("CANDIDATE2");
        frame.setBackground(Color.red);
        frame.setLayout(new BorderLayout(50, 50));
        frame.add(button1, BorderLayout.NORTH);
        frame.add(button2, BorderLayout.SOUTH);
        frame.add(button3, BorderLayout.EAST);
        frame.add(button4, BorderLayout.WEST);
        frame.add(button5, BorderLayout.CENTER);

        frame.setVisible(true);

      button4.addActionListener(new ActionListener() {
          @Override
          public void actionPerformed(ActionEvent e) {
              if(e.getSource() == button4) {
                  System.out.println("YOU voted for CANDIDATE1");
                  frame.setVisible(false);
                  frame.dispose();
              }
          }
      });
      button5.addActionListener(new ActionListener() {
          @Override
          public void actionPerformed(ActionEvent e) {
              if(e.getSource() == button5){
                  System.out.println("YOU voted for CANDIDATE2");
                  frame.setVisible(false);
                  frame.dispose();
              }
          }
      });
        button3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(e.getSource() == button3) {
                    System.out.println("YOU voted for CANDIDATE3");
                    frame.setVisible(false);
                    frame.dispose();
                }
            }
        });
    }

}
