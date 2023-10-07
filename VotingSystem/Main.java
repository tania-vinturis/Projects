package com.company;

import javax.swing.*;
import java.awt.*;
import java.io.FileReader;
import java.io.PrintStream;
import java.net.SocketOption;
import java.time.LocalDate;
import java.util.Random;
import java.util.Scanner;
import java.lang.String;
import java.io.IOException;


public class Main extends GUI{


    public static void main(String[] args) {
        Scanner textBox = new Scanner(System.in);
        System.out.println("Do you want to vote?");
        System.out.println("1. Yes");
        System.out.println("2. No");
        System.out.println("Enter your response:");
        String response = textBox.nextLine();

        if(response.equals("yes") || response.equals("Yes") || response.equals("YES")){
            System.out.println("Thank you for your interest!");
            System.out.println("Enter your name:");
            String name = textBox.nextLine();
            System.out.println("Enter your age:");
           int age = textBox.nextInt();
            if(age < 18){
                System.out.println("We are sorry to inform you that you cannot vote until your age is 18.");
            }
            else{
                new GUI();
            }
        }

    }
}
