#ifndef INTERACTIONS_H
#define INTERACTIONS_H

class Interactions {
  public:
    static void keyboard(unsigned char key, int x, int y);

    static void handleSpecialKeypress(int key, int x, int y);

    static void mouseMove(int x, int y);

    static void mouseDrag(int x, int y);

    static void reshape(int x, int y);

    static void animation();

    static void exitfunc();

    static void printInstructions();
};

#endif /* ifndef INTERACTIONS_H */
